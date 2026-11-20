#!/usr/bin/env python3
"""Train Phase 5 VEC policies with held-out validation checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vec.evaluation import (  # noqa: E402
    ScenarioManifest,
    create_standard_manifest,
)
from src.vec.training import TrainingConfig, VecTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("maddpg", "domain_randomized", "reptile"),
        default="domain_randomized",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scenarios-per-epoch", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--drain", type=int, default=100)
    parser.add_argument("--train-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--validation-seeds", default="100,101,102,103,104")
    parser.add_argument(
        "--test-seeds", default="200,201,202,203,204,205,206,207,208,209"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-queue-weight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny integration check, not a research training run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest:
        manifest = ScenarioManifest.load(args.manifest)
    elif args.smoke:
        manifest = create_standard_manifest(
            {"train": (0,), "validation": (100,), "test": (200,)},
            horizon_slots=3,
            drain_slots=1,
            manifest_id="training-smoke-v1",
        )
    else:
        manifest = create_standard_manifest(
            {
                "train": _seeds(args.train_seeds),
                "validation": _seeds(args.validation_seeds),
                "test": _seeds(args.test_seeds),
            },
            horizon_slots=args.horizon,
            drain_slots=args.drain,
        )
    config = TrainingConfig(
        mode=args.mode,
        epochs=args.epochs,
        scenarios_per_epoch=args.scenarios_per_epoch,
        seed=args.seed,
        device=args.device,
        no_queue_weight=args.no_queue_weight,
    )
    if args.smoke:
        config = replace(
            config,
            epochs=min(config.epochs, 1),
            scenarios_per_epoch=1,
            validation_scenarios=1,
            actor_hidden_dim=16,
            critic_hidden_dim=24,
            batch_size=2,
            replay_capacity=32,
            warmup_steps=2,
            reptile_inner_updates=1,
        )
    result = VecTrainer(config, manifest, args.output).train(resume=args.resume)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def _seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("each scenario split requires at least one seed")
    return seeds


if __name__ == "__main__":
    main()
