"""Portable deterministic training, validation and exact optimizer/RNG resume."""

import json
import os
import platform
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from . import __version__
from .checkpoint import restore, save_resume, save_weights
from .data import Data, random_inputs
from .evaluate import evaluate
from .flow import convert
from .model import ActionDiT
from .steps import make_optimizer, make_step
from .utils import append_json, dump, read_json, sha

DEFAULTS = dict(
    seed=47,
    steps=18000,
    batch_size=8,
    learning_rate=0.0001,
    warmup_steps=200,
    eval_every=1000,
    eval_windows=64,
    samples=8,
    noise_seed=747,
    checkpoint_every=250,
    compiled=True,
)


def contract(config, data_dir):
    package = Path(__file__).parent
    return {
        "config": config,
        "dataset_sha256": sha(Path(data_dir) / "manifest.json"),
        "source_sha256": {p.name: sha(p) for p in sorted(package.glob("*.py"))},
        "version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mlx": mx.__version__,
    }


def read_config(path):
    supplied = read_json(path) if path else {}
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    config = {**DEFAULTS, **supplied}
    for key in ("steps", "batch_size", "eval_every", "eval_windows", "samples", "checkpoint_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["warmup_steps"] < 0 or config["learning_rate"] <= 0:
        raise ValueError("invalid warmup_steps or learning_rate")
    return config


def publish_best(output, record):
    """Expose only the best model referenced by a committed resume record."""
    if record is None:
        for name in ("best.safetensors", "best.json"):
            (output / name).unlink(missing_ok=True)
        return
    source = output / record["file"]
    if sha(source) != record["sha256"]:
        raise ValueError("best checkpoint integrity failed")
    temporary = output / "best.partial.safetensors"
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, output / "best.safetensors")
    dump(output / "best.json", record)


def train(data_dir, output, config_path=None, resume=False, stop_after=None):
    config = read_config(config_path)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop-after must be positive")
    output = Path(output)
    train_data, validation = Data(data_dir, "train"), Data(data_dir, "validation")
    identity = contract(config, data_dir)
    if resume:
        old = read_json(output / "contract.json")
        if old != identity:
            raise ValueError("resume contract differs: code, config, data or runtime changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        dump(output / "contract.json", identity)
    mx.random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = ActionDiT()
    mx.eval(model.parameters())
    optimizer = make_optimizer(model)
    start, best, best_record = 0, None, None
    if resume:
        metadata = restore(model, optimizer, output)
        start, best = metadata["step"], metadata["best_rmse"]
        rng.bit_generator.state = metadata["rng"]
        best_record = metadata.get("best_record")
        publish_best(output, best_record)
        # A crash can leave log rows beyond the committed checkpoint. Truncate those rows.
        for name in ("training.jsonl", "validation.jsonl"):
            p = output / name
            if p.exists():
                rows = [json.loads(s) for s in p.read_text().splitlines()]
                p.write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= start))
    step_fn = make_step(model, optimizer, compiled=config["compiled"])
    finish = min(config["steps"], stop_after if stop_after is not None else config["steps"])
    if start >= finish:
        raise ValueError("checkpoint already reached requested stop step")

    def checkpoint(step):
        save_resume(
            model,
            optimizer,
            output,
            {
                "step": step,
                "rng": rng.bit_generator.state,
                "best_rmse": best,
                "best_record": best_record,
            },
        )

        publish_best(output, best_record)
        for old_best in output.glob("best_step_*.safetensors"):
            if best_record is None or old_best.name != best_record["file"]:
                old_best.unlink()

    for step in range(start + 1, finish + 1):
        optimizer.learning_rate = config["learning_rate"] * min(
            1.0, step / max(1, config["warmup_steps"])
        )
        ids = rng.integers(len(train_data), size=config["batch_size"])
        noise, tau = random_inputs(rng, len(ids))
        loss, parts, grad = step_fn(convert(train_data.batch(ids)), mx.array(noise), mx.array(tau))
        if not np.isfinite([float(loss), float(grad)]).all():
            raise FloatingPointError(
                "nonfinite training loss or gradient; last committed checkpoint retained"
            )
        append_json(
            output / "training.jsonl",
            {"step": step, "loss": float(loss), "gradient_norm": float(grad)},
        )
        if step % config["eval_every"] == 0 or step == config["steps"]:
            result = evaluate(
                model,
                validation,
                validation.panel(config["eval_windows"]),
                samples=config["samples"],
                noise_seed=config["noise_seed"],
            )
            append_json(output / "validation.jsonl", {"step": step, **result})
            if best is None or result["sample_rmse_m"] < best:
                best = result["sample_rmse_m"]
                best_file = output / f"best_step_{step}.safetensors"
                save_weights(model, best_file)
                best_record = {
                    "step": step,
                    "metric": "sample_rmse_m",
                    "value": best,
                    "file": best_file.name,
                    "sha256": sha(best_file),
                }
            print(
                json.dumps({"step": step, "validation_rmse_m": result["sample_rmse_m"]}), flush=True
            )
        if (
            step % config["checkpoint_every"] == 0
            or step % config["eval_every"] == 0
            or step == finish
        ):
            checkpoint(step)
    summary = {
        "step": finish,
        "target_steps": config["steps"],
        "best_validation_rmse_m": best,
        "parameters": sum(v.size for _, v in tree_flatten(model.parameters())),
        "synthetic": bool(train_data.manifest.get("synthetic_data", False))
        or not train_data.manifest["encoder"]["scientific_use"],
    }
    dump(output / "summary.json", summary)
    return summary


def score(data_dir, weights, output, samples=8, noise_seed=747, max_windows=None):
    data = Data(data_dir, "validation")
    model = ActionDiT()
    model.load_weights(str(weights), strict=True)
    mx.eval(model.parameters())
    result = evaluate(
        model, data, data.panel(max_windows or len(data)), samples=samples, noise_seed=noise_seed
    )
    result.update(
        dataset_sha256=sha(Path(data_dir) / "manifest.json"),
        weights_sha256=sha(weights),
        synthetic=bool(data.manifest.get("synthetic_data", False))
        or not data.manifest["encoder"]["scientific_use"],
        split="validation",
    )
    dump(output, result)
    return result
