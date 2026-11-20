#!/usr/bin/env python3
"""Run the complete multi-seed VEC training and paired evaluation study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vec.experiments import StudyConfig, VecStudyRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learner-seeds", default="0,1,2,3,4")
    parser.add_argument("--train-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--validation-seeds", default="100,101,102,103,104")
    parser.add_argument(
        "--test-seeds", default="200,201,202,203,204,205,206,207,208,209"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scenarios-per-epoch", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--drain", type=int, default=100)
    parser.add_argument("--genetic-budget", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent training/evaluation subprocesses to run concurrently",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StudyConfig(
        learner_seeds=_seeds(args.learner_seeds),
        train_seeds=_seeds(args.train_seeds),
        validation_seeds=_seeds(args.validation_seeds),
        test_seeds=_seeds(args.test_seeds),
        epochs=args.epochs,
        scenarios_per_epoch=args.scenarios_per_epoch,
        horizon_slots=args.horizon,
        drain_slots=args.drain,
        genetic_budget=args.genetic_budget,
        device=args.device,
        smoke=args.smoke,
    )
    runner = VecStudyRunner(config, args.output, project_root=PROJECT_ROOT)
    runner.run(
        dry_run=args.dry_run,
        skip_evaluation=args.skip_evaluation,
        plots=not args.no_plots,
        workers=args.workers,
    )
    print(runner.output / "study_status.json")
    return 0


def _seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("seed lists cannot be empty")
    return seeds


if __name__ == "__main__":
    raise SystemExit(main())
