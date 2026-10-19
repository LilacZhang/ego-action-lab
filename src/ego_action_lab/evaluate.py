"""Masked paired metrics, exact native PTS and expected sample error."""

from collections import defaultdict
from functools import partial

import mlx.core as mx
import numpy as np

from .data import random_inputs
from .flow import convert, loss_fn, reference
from .metrics import angle_degrees, paired_session_interval


def evaluate(model, data, indices, samples=8, noise_seed=747, window_batch=4):
    if not len(indices) or samples < 1 or window_batch < 1:
        raise ValueError("evaluation requires windows, samples and batch size > 0")
    norm = data.normalization
    scale = np.array(norm["scale"])
    mean = np.array(norm["mean"])
    rng = np.random.default_rng(noise_seed)
    frng = np.random.default_rng(noise_seed + 100000)
    # Preserve the original eight-window RNG grouping, independent of inference microbatch.
    fn_bank = []
    ft_bank = []
    gn_bank = []
    for i in range(0, len(indices), 8):
        size = min(8, len(indices) - i)
        fn, ft = random_inputs(frng, size)
        gn, _ = random_inputs(rng, size * samples)
        fn_bank.append(fn)
        ft_bank.append(ft)
        gn_bank.append(gn.reshape(size, samples, 16, 48))
    fn_bank = np.concatenate(fn_bank)
    ft_bank = np.concatenate(ft_bank)
    gn_bank = np.concatenate(gn_bank)
    ids_pos = np.array([0, 1, 2, 24, 25, 26])
    sse = np.zeros((4, 16))
    count_h = np.zeros(16)
    fm = []
    parts = []
    rows = []
    strata = defaultdict(lambda: np.zeros(4))
    body_sum = defaultdict(float)
    body_count = defaultdict(int)

    # Read-only compile: current parameters are explicit captured inputs.
    @partial(mx.compile, inputs=[model.state])
    def velocity(b, action, tau):
        return model({**b, "noisy": action, "tau": tau})

    def accumulate(key, v, m, square=True):
        v, m = np.broadcast_arrays(v, m)
        body_sum[key] += float(np.where(m, v * v if square else v, 0).sum())
        body_count[key] += int(m.sum())

    for start in range(0, len(indices), window_batch):
        ids = indices[start : start + window_batch]
        a = data.batch(ids)
        b = convert(a)
        n = len(ids)
        noise, tau = fn_bank[start : start + n], ft_bank[start : start + n]
        loss, components = loss_fn(model, b, mx.array(noise), mx.array(tau))
        mx.eval(loss, components)
        fm.extend([float(loss)] * n)
        parts.extend([np.array(components)] * n)
        cond = convert(
            {k: np.repeat(a[k], samples, axis=0) for k in ["image", "state", "state_mask"]}
        )
        noise = gn_bank[start : start + n].reshape(n * samples, 16, 48)
        action = mx.array(noise)
        for tick in range(5):
            action = (
                action
                + velocity(cond, action, mx.full((n * samples,), tick / 5, dtype=mx.float32)) / 5
            )
            mx.eval(action)
        action = action + reference(cond)
        mx.eval(action)
        generated = np.array(action).reshape(n, samples, 16, 48)
        truth = a["action"]
        current = a["state"]
        mask = a["action_mask"] & a["state_mask"][:, None]
        m = mask[:, :, ids_pos]
        err = (generated[:, :, :, ids_pos] - truth[:, None, :, ids_pos]) * scale[ids_pos]
        arrays = [
            (err**2).mean(1),
            ((generated.mean(1)[:, :, ids_pos] - truth[:, :, ids_pos]) * scale[ids_pos]) ** 2,
            ((current[:, None, ids_pos] - truth[:, :, ids_pos]) * scale[ids_pos]) ** 2,
            err[:, 0] ** 2,
        ]
        for k, arr in enumerate(arrays):
            sse[k] += np.where(m, arr, 0).sum((0, 2))
        count_h += m.sum((0, 2))
        for j, index in enumerate(ids):
            ei, w = data.index[int(index)]
            e = data.episodes[ei]
            c = int(m[j].sum())
            sums = [float(np.where(m[j], v[j], 0).sum()) for v in arrays]
            rows.append(
                {
                    "index": int(index),
                    "session": str(e["session"]),
                    "count": c,
                    "sample_sse": sums[0],
                    "mean_sse": sums[1],
                    "hold_sse": sums[2],
                }
            )
            hold = np.sqrt(sums[2] / max(c, 1))
            key = (
                "hold_rmse_lt_2cm"
                if hold < 0.02
                else "hold_rmse_2_to_5cm"
                if hold < 0.05
                else "hold_rmse_ge_5cm"
            )
            strata[key] += sums[:3] + [c]
            for side, sl in [("left", slice(0, 3)), ("right", slice(3, 6))]:
                strata[side] += [
                    float(np.where(m[j, :, sl], v[j, :, sl], 0).sum()) for v in arrays[:3]
                ] + [int(m[j, :, sl].sum())]
        raw = generated * scale + mean
        gt = truth * scale + mean
        cur = current * scale + mean
        for side, s in [("left", 0), ("right", 24)]:
            for off, name in [(s, "wrist")] + [(s + 9 + 3 * j, f"tip{j}") for j in range(5)]:
                sl = slice(off, off + 3)
                mm = mask[:, :, sl].all(-1)
                for variant, pred in [
                    ("sample", raw),
                    ("mean", raw.mean(1)[:, None]),
                    ("hold", cur[:, None, None]),
                ]:
                    accumulate(
                        f"{side}_{name}_{variant}_3d_rmse_m",
                        np.linalg.norm(pred[..., sl] - gt[:, None, :, sl], axis=-1),
                        mm[:, None],
                    )
            sl = slice(s + 3, s + 9)
            mm = mask[:, :, sl].all(-1)
            for variant, pred in [
                ("sample", raw),
                ("mean", raw.mean(1)[:, None]),
                ("hold", cur[:, None, None]),
            ]:
                angle, _ = angle_degrees(pred[..., sl], gt[:, None, :, sl])
                accumulate(f"{side}_rotation_{variant}_mean_degrees", angle, mm[:, None], False)
            dt = []
            native = []
            for index in ids:
                ei, w = data.index[int(index)]
                e = data.episodes[ei]
                dt.append(np.diff(e["pts"][e["frame_indices"][w]]))
                native.append(e["masks"][w, :, s : s + 3].all(-1))
            dt = np.stack(dt)
            native = np.stack(native)
            valid = native[:, 1:] & native[:, :-1] & a["state_mask"][:, s : s + 3].all(-1)[:, None]
            gp = np.concatenate([cur[:, None, s : s + 3], gt[:, :, s : s + 3]], axis=1)
            gv = np.diff(gp, axis=1) / dt[..., None]
            for variant, pred in [
                ("sample", raw),
                ("mean", raw.mean(1)[:, None]),
                ("hold", np.broadcast_to(cur[:, None, None], (n, 1, 16, 48))),
            ]:
                pp = np.concatenate(
                    [
                        np.broadcast_to(cur[:, None, None, s : s + 3], (n, pred.shape[1], 1, 3)),
                        pred[..., s : s + 3],
                    ],
                    axis=2,
                )
                pv = np.diff(pp, axis=2) / dt[:, None, :, None]
                accumulate(
                    f"{side}_wrist_velocity_{variant}_3d_rmse_m_s",
                    np.linalg.norm(pv - gv[:, None], axis=-1),
                    valid[:, None],
                )
    count = int(count_h.sum())
    rmse = np.sqrt(sse.sum(1) / max(count, 1))
    pc = np.mean(parts, axis=0)
    return {
        "windows": len(indices),
        "samples": samples,
        "euler_steps": 5,
        "noise_seed": noise_seed,
        "valid_position_scalars": count,
        "fm_family_loss": float(np.mean(fm)),
        "fm_scalar_loss": float(pc[3]),
        "fm_wrist_position": float(pc[0]),
        "fm_wrist_rotation": float(pc[1]),
        "fm_fingertips": float(pc[2]),
        "sample_rmse_m": float(rmse[0]),
        "mean_rmse_m": float(rmse[1]),
        "hold_rmse_m": float(rmse[2]),
        "first_sample_rmse_m": float(rmse[3]),
        "sample_to_hold_ratio": float(rmse[0] / rmse[2]) if rmse[2] > 0 else None,
        "mean_to_hold_ratio": float(rmse[1] / rmse[2]) if rmse[2] > 0 else None,
        "sample_3d_position_rmse_m": float(np.sqrt(3) * rmse[0]),
        "by_horizon": {
            "requested_seconds": (np.arange(1, 17) * 0.05).tolist(),
            **{
                key: np.sqrt(sse[j] / np.maximum(count_h, 1)).tolist()
                for j, key in enumerate(
                    ["sample_rmse_m", "mean_rmse_m", "hold_rmse_m", "first_rmse_m"]
                )
            },
        },
        "strata": {
            key: {
                "sample_rmse_m": float(np.sqrt(v[0] / max(v[3], 1))),
                "mean_rmse_m": float(np.sqrt(v[1] / max(v[3], 1))),
                "hold_rmse_m": float(np.sqrt(v[2] / max(v[3], 1))),
                "valid_scalars": int(v[3]),
            }
            for key, v in strata.items()
        },
        "body_metrics": {
            k: None
            if body_count[k] == 0
            else float(v / body_count[k])
            if k.endswith("mean_degrees")
            else float(np.sqrt(v / body_count[k]))
            for k, v in body_sum.items()
        },
        "paired_session_bootstrap": paired_session_interval(rows),
        "metric": "paired valid wrist coordinate RMSE in metres; raw tracking labels",
        "body_counts": dict(body_count),
    }
