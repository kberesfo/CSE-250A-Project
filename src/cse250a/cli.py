"""Experiment runner: `uv run cse250a run --model hmm --seed 1`."""

import argparse
from datetime import datetime
from pathlib import Path

from cse250a.modules.config import config


def run(model: str, name: str | None) -> Path:
    """Create a run directory and save the resolved config alongside its outputs."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.paths.results_dir / f"{name or model}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    # TODO: dispatch to the model once one exists, writing outputs into run_dir.
    print(f"model={model} seed={config.seed} -> {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="cse250a")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run an experiment and save results")
    run_p.add_argument("--model", required=True, help="model to run, e.g. hmm or bn")
    run_p.add_argument("--seed", type=int, help="overrides SEED / config.seed")
    run_p.add_argument("--name", help="run directory prefix (default: model name)")

    args = parser.parse_args()
    if args.seed is not None:
        config.seed = args.seed
    if args.command == "run":
        run(args.model, args.name)


if __name__ == "__main__":
    main()
