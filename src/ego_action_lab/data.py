"""Prepared dataset contract; no original workspace paths or remote services."""

from pathlib import Path

import numpy as np

from .utils import read_json, sha


def random_inputs(rng, n):
    return (
        rng.standard_normal((n, 16, 48)).astype(np.float32),
        rng.uniform(0, 0.999, n).astype(np.float32),
    )


class Data:
    def __init__(self, directory, split, verify=True):
        self.root = Path(directory)
        manifest = read_json(self.root / "manifest.json")
        if manifest["status"] != "complete" or split not in ("train", "validation"):
            raise ValueError("requires a complete train/validation dataset")
        if verify:
            for name, expected in manifest["sha256"].items():
                if sha(self.root / name) != expected:
                    raise ValueError(f"dataset integrity check failed: {name}")
        self.manifest = manifest
        self.normalization = read_json(self.root / "normalization.json")
        self.episodes, self.index = [], []
        for name in manifest["splits"][split]:
            with np.load(self.root / name, allow_pickle=False) as arrays:
                e = {key: arrays[key] for key in arrays.files}
            self.index.extend((len(self.episodes), j) for j in range(len(e["values"])))
            self.episodes.append(e)
        self.features = np.load(self.root / f"{split}_features.npy", mmap_mode="r")
        self.feature_rows = np.load(self.root / f"{split}_window_rows.npy")
        if not self.index or len(self.feature_rows) != len(self.index):
            raise ValueError("empty or inconsistent dataset")
        if self.features.shape[1:] != (196, 384):
            raise ValueError("expected 196 x 384 image features")

    def __len__(self):
        return len(self.index)

    def panel(self, n):
        return np.linspace(0, len(self) - 1, min(n, len(self)), dtype=np.int64)

    def batch(self, indices):
        pairs = [self.index[int(i)] for i in indices]
        values = np.stack([self.episodes[e]["values"][w] for e, w in pairs])
        masks = np.stack([self.episodes[e]["masks"][w] for e, w in pairs])
        if not np.isfinite(values).all() or not masks[:, 1:].any(axis=(1, 2)).all():
            raise ValueError("invalid prepared values/masks")
        return {
            "state": values[:, 0],
            "state_mask": masks[:, 0],
            "action": values[:, 1:],
            "action_mask": masks[:, 1:],
            "image": np.asarray(self.features[self.feature_rows[indices]], np.float32),
        }
