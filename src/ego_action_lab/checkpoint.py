"""Atomic array checkpoints, including optimizer and external RNG states."""

import json
import os
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .utils import dump, sha


def save_weights(model, path):
    path = Path(path)
    tmp = path.with_name(path.stem + ".partial.safetensors")
    model.save_weights(str(tmp))
    os.replace(tmp, path)


def save_resume(model, optimizer, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Two generations: metadata is the commit record and points to completed files.
    if (directory / "resume.json").exists():
        previous = json.loads((directory / "resume.json").read_text())
        slot = 1 - int(Path(previous["weights_file"]).stem.rsplit("_", 1)[1])
    else:
        slot = 0
    weights = directory / f"resume_{slot}.safetensors"
    state = directory / f"optimizer_{slot}.safetensors"
    save_weights(model, weights)
    tmp = state.with_name(state.stem + ".partial.safetensors")
    mx.eval(optimizer.state)
    mx.save_safetensors(str(tmp), dict(tree_flatten(optimizer.state)))
    os.replace(tmp, state)
    dump(
        directory / "resume.json",
        {
            **metadata,
            "weights_file": weights.name,
            "optimizer_file": state.name,
            "weights_sha256": sha(weights),
            "optimizer_sha256": sha(state),
        },
    )


def restore(model, optimizer, directory):
    directory = Path(directory)
    meta = json.loads((directory / "resume.json").read_text())
    for key in ("weights", "optimizer"):
        if sha(directory / meta[key + "_file"]) != meta[key + "_sha256"]:
            raise ValueError(f"checkpoint integrity failed: {key}")
    model.load_weights(str(directory / meta["weights_file"]), strict=True)
    optimizer.state = tree_unflatten(list(mx.load(str(directory / meta["optimizer_file"])).items()))
    mx.eval(model.parameters(), optimizer.state)
    return meta
