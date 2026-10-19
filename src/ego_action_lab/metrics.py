"""Pure NumPy metrics; preserve original session bootstrap and plot definitions."""

from collections import defaultdict

import numpy as np


def paired_session_interval(rows, repetitions=2000, seed=91018):
    """Resample sessions, retaining all correlated windows inside each session."""
    totals = defaultdict(lambda: np.zeros(3, np.float64))
    for row in rows:
        totals[row["session"]] += [row["sample_sse"], row["hold_sse"], row["count"]]
    if not totals or sum(v[1] for v in totals.values()) <= 0:
        return {"session_count": len(totals), "sample_to_hold_rmse_ratio_ci95": None}
    array = np.stack(list(totals.values()))
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(len(array), size=(repetitions, len(array)))].sum(1)
    sampled = sampled[sampled[:, 1] > 0]
    ratio = np.sqrt(sampled[:, 0] / sampled[:, 1])
    return {
        "session_count": len(array),
        "replicates": repetitions,
        "sample_to_hold_rmse_ratio_ci95": np.quantile(ratio, [0.025, 0.975]).tolist(),
        "unit": "session; windows retained together; paired predictors",
    }


def rotation_matrix(columns):
    a, b = (columns[..., :3], columns[..., 3:])
    an = np.linalg.norm(a, axis=-1, keepdims=True)
    u = a / np.maximum(an, 1e-08)
    residual = b - (u * b).sum(-1, keepdims=True) * u
    bn = np.linalg.norm(residual, axis=-1, keepdims=True)
    v = residual / np.maximum(bn, 1e-08)
    return (np.stack([u, v, np.cross(u, v)], axis=-1), (an[..., 0] > 1e-05) & (bn[..., 0] > 1e-05))


def angle_degrees(predicted, target):
    r, valid = rotation_matrix(predicted)
    gt, _ = rotation_matrix(target)
    cosine = (np.einsum("...ij,...ij->...", r, gt) - 1) / 2
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return (np.where(valid, angle, 180.0), valid)
