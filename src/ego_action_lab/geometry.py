"""EgoDex part-wise geometry masks and safe masked flow matching."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

FINGERS = ["ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip"]


def se3_valid(matrix):
    matrix = np.asarray(matrix)
    finite = np.isfinite(matrix).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], matrix, np.eye(4))
    rotation = safe[..., :3, :3]
    return (
        finite
        & (np.max(np.abs(safe[..., 3, :] - np.array([0, 0, 0, 1])), axis=-1) <= 1e-4)
        & (
            np.linalg.norm(np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3), axis=(-2, -1))
            <= 1e-2
        )
        & (np.abs(np.linalg.det(rotation) - 1) <= 1e-2)
    )


class RawEpisode:
    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(path, "r") as f:
            self.session = str(f.attrs.get("session_name", ""))
            self.camera = np.asarray(f["transforms/camera"], dtype=np.float64)
            self.n = len(self.camera)
            self.transform, self.confidence = {}, {}
            for side in ["left", "right"]:
                for part in ["Hand"] + FINGERS:
                    name = side + part
                    self.transform[name] = (
                        np.asarray(f["transforms"][name], dtype=np.float64)
                        if name in f["transforms"]
                        else np.full((self.n, 4, 4), np.nan)
                    )
                    self.confidence[name] = (
                        np.asarray(f["confidences"][name], dtype=np.float64)
                        if "confidences" in f and name in f["confidences"]
                        else np.full(self.n, np.nan)
                    )
                    if self.transform[name].shape != (self.n, 4, 4) or self.confidence[
                        name
                    ].shape != (self.n,):
                        raise ValueError(f"inconsistent native frame length: {name}")
        self.camera_valid = se3_valid(self.camera)
        self.mask = np.zeros((self.n, 48), dtype=bool)
        for side, start in [("left", 0), ("right", 24)]:
            c = self.confidence[side + "Hand"]
            wrist = se3_valid(self.transform[side + "Hand"]) & np.isfinite(c) & (c >= 0.5)
            self.mask[:, start : start + 9] = wrist[:, None]
            for j, finger in enumerate(FINGERS):
                tr, c = self.transform[side + finger], self.confidence[side + finger]
                valid = (
                    wrist
                    & np.isfinite(tr[:, :3, 3]).all(axis=-1)
                    & np.isfinite(tr[:, 3, :]).all(axis=-1)
                    & (np.max(np.abs(tr[:, 3, :] - [0, 0, 0, 1]), axis=-1) <= 1e-4)
                    & np.isfinite(c)
                    & (c >= 0.5)
                )
                self.mask[:, start + 9 + 3 * j : start + 12 + 3 * j] = valid[:, None]

    def window(self, indices, supported=None):
        indices = np.asarray(indices, dtype=np.int64)
        if not self.camera_valid[indices[0]]:
            raise ValueError("invalid camera anchor")
        anchor = self.camera[indices[0]]
        rotation, origin = anchor[:3, :3], anchor[:3, 3]
        value = np.full((len(indices), 48), np.nan, dtype=np.float32)
        mask = self.mask[indices].copy()
        if supported is not None:
            mask &= np.asarray(supported, bool)[None]
        for side, start in [("left", 0), ("right", 24)]:
            tf = self.transform[side + "Hand"][indices]
            value[:, start : start + 3] = (tf[:, :3, 3] - origin) @ rotation
            rot = np.einsum("ij,njk->nik", rotation.T, tf[:, :3, :3])
            value[:, start + 3 : start + 9] = np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=-1)
            for j, finger in enumerate(FINGERS):
                pos = self.transform[side + finger][indices, :3, 3]
                value[:, start + 9 + j * 3 : start + 12 + j * 3] = (pos - origin) @ rotation
        mask &= np.isfinite(value)
        value[~mask] = np.nan
        return value, mask


def eligible(mask):
    return any(
        mask[0, s : s + 9].all() and mask[1:, s : s + 9].all(axis=-1).sum() >= 13 for s in (0, 24)
    )


def normalize(value, mask, mean, scale):
    mask = np.asarray(mask, bool)
    if not np.isfinite(np.asarray(value)[mask]).all():
        raise ValueError("nonfinite value marked valid")
    safe = np.where(mask, value, np.asarray(mean))
    return np.where(mask, (safe - mean) / scale, 0).astype(np.float32)
