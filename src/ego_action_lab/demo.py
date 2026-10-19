"""Generate toy HDF5/video pairs locally; no real people or dataset downloads."""

from pathlib import Path

import av
import h5py
import numpy as np

from .geometry import FINGERS
from .prepare import prepare
from .utils import dump


def demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for split, offset in [("train", 0.0), ("validation", 0.1)]:
        n = 36
        t = np.arange(n) / 20
        identity = np.tile(np.eye(4), (n, 1, 1))
        with h5py.File(output / f"{split}.hdf5", "w") as f:
            f.attrs["session_name"] = f"toy-{split}"
            f["transforms/camera"] = identity
            for side, sign in [("left", -1), ("right", 1)]:
                for j, part in enumerate(["Hand"] + FINGERS):
                    matrix = identity.copy()
                    matrix[:, 0, 3] = sign * 0.15 + 0.04 * np.sin(2 * t + offset) + j * 0.005
                    matrix[:, 1, 3] = 0.03 * np.cos(t) + offset
                    matrix[:, 2, 3] = 0.5 + j * 0.01
                    f[f"transforms/{side}{part}"] = matrix
                    confidence = np.ones(n)
                    if part == "LittleFingerTip":
                        confidence[::5] = 0  # Exercise partial observations.
                    f[f"confidences/{side}{part}"] = confidence
        with av.open(str(output / f"{split}.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=20)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            for idx in range(n):
                rgb = np.zeros((48, 64, 3), np.uint8)
                rgb[:, :, 0] = idx * 5
                rgb[:, :, 1] = 40 if split == "train" else 100
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        records.append(
            {
                "split": split,
                "session": f"toy-{split}",
                "hdf5": f"{split}.hdf5",
                "video": f"{split}.mp4",
            }
        )
    dump(output / "episodes.json", {"episodes": records, "synthetic": True})
    prepare(output / "episodes.json", output / "prepared", stride=5, synthetic=True)
    return {
        "dataset": str(output / "prepared"),
        "synthetic": True,
        "note": "pipeline test only; no research result",
    }
