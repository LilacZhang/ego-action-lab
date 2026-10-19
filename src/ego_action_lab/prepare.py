"""Build session-disjoint windows, train-only statistics and current-frame features."""

import shutil
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .geometry import RawEpisode, eligible, normalize
from .utils import dump, read_json, sha


def timestamps(video):
    with av.open(str(video)) as container:
        times = [
            float(f.pts * f.time_base) if f.pts is not None else np.nan
            for f in container.decode(video=0)
        ]
    pts = np.asarray(times, np.float64)
    if not len(pts) or not np.isfinite(pts).all() or not (np.diff(pts) > 0).all():
        raise ValueError("video must have finite, strictly increasing native PTS")
    return pts


def windows(raw, pts, stride=10, tolerance=0.025):
    if len(pts) != raw.n:
        raise ValueError("video frames and HDF5 transforms must align one-to-one")
    values, masks, frames = [], [], []
    for anchor in range(0, len(pts), stride):
        if not raw.camera_valid[anchor]:
            continue
        wanted = pts[anchor] + np.arange(17) * 0.05
        right = np.clip(np.searchsorted(pts, wanted), 0, len(pts) - 1)
        left = np.maximum(right - 1, 0)
        ids = np.where(np.abs(pts[left] - wanted) < np.abs(pts[right] - wanted), left, right)
        if np.max(np.abs(pts[ids] - wanted)) > tolerance or not (np.diff(ids) > 0).all():
            continue
        v, m = raw.window(ids)
        if eligible(m):
            values.append(v)
            masks.append(m)
            frames.append(ids)
    if not values:
        raise ValueError("episode has no eligible 0.8-second windows")
    return np.stack(values), np.stack(masks), np.stack(frames)


def fit_normalization(episodes):
    count = np.zeros(48, np.int64)
    total = np.zeros(48)
    square = np.zeros(48)
    for e in episodes:
        m, v = e["masks"], e["values"].astype(np.float64)
        safe = np.where(m, v, 0)
        count += m.sum((0, 1))
        total += safe.sum((0, 1))
        square += (safe * safe).sum((0, 1))
    mean = total / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean * mean, 0))
    # Constant but observed coordinates remain supported; unseen coordinates are masked globally.
    scale = np.where(scale > 1e-6, scale, 1.0)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "count": count.tolist(),
        "supported": (count > 0).tolist(),
        "fit_split": "train",
        "unit": "masked window coordinates",
    }


def resize224(rgb):
    image = Image.fromarray(rgb)
    factor = 224 / max(image.size)
    w, h = (max(1, round(x * factor)) for x in image.size)
    image = image.resize((w, h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224))
    canvas.paste(image, ((224 - w) // 2, (224 - h) // 2))
    return np.asarray(canvas)


def encode_frames(video, ids, pts, encoder):
    needed = set(map(int, ids))
    found = {}
    with av.open(str(video)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in needed:
                if frame.pts is None or abs(float(frame.pts * frame.time_base) - pts[idx]) > 1e-6:
                    raise ValueError("native PTS changed during preparation")
                rgb = resize224(frame.to_ndarray(format="rgb24"))
                feature = encoder(rgb)
                if feature.shape != (196, 384) or not np.isfinite(feature).all():
                    raise ValueError("bad image features")
                found[idx] = feature.astype(np.float16)
            if idx >= max(needed):
                break
    if set(found) != needed:
        raise ValueError("missing requested current frame")
    return np.stack([found[int(i)] for i in ids])


def prepare(manifest_path, output, weights=None, stride=10, synthetic=False):
    if stride < 1:
        raise ValueError("stride must be positive")
    output = Path(output).resolve()
    source = Path(manifest_path).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    records = read_json(source)["episodes"]
    sessions = {}
    seen_paths = set()
    seen_hashes = set()
    episodes = []
    provenance = []
    for record in records:
        split, session = record["split"], str(record["session"]).strip()
        if split not in ("train", "validation") or not session:
            raise ValueError("every episode needs train/validation and a nonempty session")
        if session in sessions and sessions[session] != split:
            raise ValueError("session leakage between train and validation")
        sessions[session] = split
        h5, video = [(source.parent / record[k]).resolve() for k in ("hdf5", "video")]
        if h5 in seen_paths or video in seen_paths:
            raise ValueError("duplicate episode source")
        seen_paths.update((h5, video))
        hashes = {"hdf5": sha(h5), "video": sha(video)}
        if any(h in seen_hashes for h in hashes.values()):
            raise ValueError("duplicate source content")
        seen_hashes.update(hashes.values())
        raw = RawEpisode(h5)
        native_session = raw.session.strip()
        if native_session and native_session != session:
            raise ValueError("manifest session disagrees with HDF5 session_name")
        pts = timestamps(video)
        v, m, ids = windows(raw, pts, stride)
        episodes.append(
            dict(
                split=split,
                session=session,
                video=video,
                pts=pts,
                values=v,
                masks=m,
                frame_indices=ids,
            )
        )
        provenance.append(dict(split=split, session=session, sha256=hashes))
    if set(sessions.values()) != {"train", "validation"}:
        raise ValueError("both train and validation sessions are required")
    norm = fit_normalization([e for e in episodes if e["split"] == "train"])
    if synthetic:
        # Smoke-test features are deliberately not a scientific visual representation.
        def encoder(rgb):
            return np.full((196, 384), rgb.mean() / 255, np.float32)

        encoder_info = {"name": "synthetic-smoke-only", "scientific_use": False}
    else:
        if weights is None:
            raise ValueError("provide timm-format DINOv3 ViT-S/16 safetensors via --weights")
        import mlx.core as mx

        from .vision import make_encoder

        encode = make_encoder(weights)

        def encoder(rgb):
            x = (
                rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
            ) / np.array([0.229, 0.224, 0.225], np.float32)
            y = encode(mx.array(x[None]))
            mx.eval(y)
            return np.array(y[0])

        encoder_info = {
            "name": "dinov3-vit-small-patch16",
            "weight_sha256": sha(weights),
            "scientific_use": True,
            "input_size": 224,
            "precision": "fp16 cache; fp32 encoder",
        }
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"incomplete preparation exists: {staging}")
    staging.mkdir(parents=True)
    try:
        dump(staging / "normalization.json", norm)
        split_files = {"train": [], "validation": []}
        for split in split_files:
            features, rows, offset = [], [], 0
            for number, e in enumerate(episodes):
                if e["split"] != split:
                    continue
                e["masks"] &= np.asarray(norm["supported"])[None, None]
                if not e["masks"][:, 1:].any((1, 2)).all():
                    raise ValueError("validation windows have no train-supported targets")
                e["values"] = normalize(e["values"], e["masks"], norm["mean"], norm["scale"])
                ids, inverse = np.unique(e["frame_indices"][:, 0], return_inverse=True)
                features.append(encode_frames(e["video"], ids, e["pts"], encoder))
                rows.extend((offset + inverse).tolist())
                offset += len(ids)
                name = f"{split}_episode_{number:04d}.npz"
                np.savez_compressed(
                    staging / name,
                    **{k: e[k] for k in ("values", "masks", "frame_indices", "pts", "session")},
                )
                split_files[split].append(name)
            np.save(staging / f"{split}_features.npy", np.concatenate(features))
            np.save(staging / f"{split}_window_rows.npy", np.array(rows, np.int64))
        hashes = {p.name: sha(p) for p in sorted(staging.iterdir())}
        dump(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "synthetic_data": bool(synthetic or read_json(source).get("synthetic", False)),
                "status": "complete",
                "splits": split_files,
                "sha256": hashes,
                "source_manifest_sha256": sha(source),
                "sources": provenance,
                "encoder": encoder_info,
                "stride_native_frames": stride,
                "horizon_seconds": 0.8,
                "target_step_seconds": 0.05,
                "max_timestamp_error_seconds": 0.025,
                "future_visual_conditions": False,
                "normalization_split": "train",
            },
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return read_json(output / "manifest.json")
