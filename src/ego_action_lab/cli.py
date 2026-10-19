"""Command-line entry point. MLX is imported only by commands that require it."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="EGO Action Lab — reproducible action experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="create a synthetic CPU pipeline example")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare", help="prepare EgoDex data and DINOv3 features (Apple Silicon)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--stride", type=int, default=10)
    p = sub.add_parser("inspect", help="verify all hashes and inspect a prepared dataset")
    p.add_argument("--data", type=Path, required=True)
    p = sub.add_parser("train", help="train or resume the 131M MLX action model")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--stop-after", type=int, help="pause at this absolute step, keeping the full config"
    )
    p = sub.add_parser("evaluate", help="score a checkpoint on validation windows")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--noise-seed", type=int, default=747)
    p.add_argument("--max-windows", type=int)
    a = parser.parse_args()
    try:
        if a.command == "demo":
            from .demo import demo

            result = demo(a.output)
        elif a.command == "prepare":
            from .prepare import prepare

            result = prepare(a.manifest, a.output, a.weights, a.stride)
        elif a.command == "inspect":
            from .data import Data

            result = {s: len(Data(a.data, s)) for s in ("train", "validation")}
        elif a.command == "train":
            from .train import train

            result = train(a.data, a.output, a.config, a.resume, a.stop_after)
        else:
            from .train import score

            if a.max_windows is not None and a.max_windows < 1:
                raise ValueError("max-windows must be positive")
            result = score(a.data, a.weights, a.output, a.samples, a.noise_seed, a.max_windows)
    except ImportError as e:
        if e.name and e.name.startswith("mlx"):
            parser.exit(
                2, 'This command requires Apple Silicon macOS and pip install -e ".[mlx]".\n'
            )
        raise
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
