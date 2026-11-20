"""Generate reproducible VEC evaluation evidence.

Example smoke run:
    python scripts/evaluate_vec.py --split test --seeds 200,201 \
        --stresses low_load,traffic_surge --horizon 50 --drain 50 \
        --methods local_only,queue_energy_greedy --output output/vec-eval
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vec.evaluation import (  # noqa: E402
    EvidenceWriter,
    EvaluationMethod,
    ScenarioManifest,
    baseline_methods,
    create_standard_manifest,
    evaluate_scenario,
    generate_figures,
    instantiate_scenario,
    missing_required_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired, censor-aware VEC evaluation scenarios."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument("--seeds", default="200,201,202,203,204")
    parser.add_argument("--stresses", default="")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--drain", type=int, default=100)
    parser.add_argument(
        "--methods",
        default=(
            "local_only,nearest_feasible_edge,queue_energy_greedy,"
            "genetic_drift,random_feasible"
        ),
    )
    parser.add_argument("--genetic-budget", type=int, default=64)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Register a learned checkpoint; MAPPO and MADDPG are detected from its metadata.",
    )
    parser.add_argument(
        "--controller-factory",
        action="append",
        default=[],
        metavar="METHOD=MODULE:CALLABLE",
        help=(
            "Register any common-interface controller, including hybrid MAPPO; "
            "the callable receives (scenario, environment)."
        ),
    )
    parser.add_argument("--adaptation-updates", type=int, default=0)
    parser.add_argument(
        "--offline-training-seconds",
        action="append",
        default=[],
        metavar="METHOD=SECONDS",
    )
    parser.add_argument("--reference-method", default="queue_energy_greedy")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete-roster", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest:
        manifest = ScenarioManifest.load(args.manifest)
    else:
        seeds = tuple(int(value) for value in args.seeds.split(",") if value)
        manifest = create_standard_manifest(
            {args.split: seeds},
            horizon_slots=args.horizon,
            drain_slots=args.drain,
            manifest_id=f"synthetic-{args.split}-cli-v1",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest.save(args.output / "scenario_manifest.json")
    if args.manifest_only:
        print(args.output / "scenario_manifest.json")
        return 0

    requested_names = {value for value in args.methods.split(",") if value}
    available = {
        method.name: method
        for method in baseline_methods(genetic_budget=args.genetic_budget)
    }
    training_times = _assignments(args.offline_training_seconds, float)
    for name, path in _assignments(args.checkpoint, str).items():
        checkpoint = str(Path(path).resolve())

        import torch

        checkpoint_metadata = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_algorithm = checkpoint_metadata.get(
            "algorithm", "hybrid_maddpg"
        )

        def factory(
            scenario,
            environment,
            *,
            method_name=name,
            checkpoint_path=checkpoint,
            algorithm=checkpoint_algorithm,
        ):
            del scenario
            if algorithm == "hybrid_mappo":
                from src.vec.learning import HybridMAPPOController

                return HybridMAPPOController.from_checkpoint(
                    method_name, checkpoint_path, environment
                )
            from src.vec.learning import HybridPolicyController

            return HybridPolicyController.from_checkpoint(
                method_name,
                checkpoint_path,
                environment,
                max_online_updates=args.adaptation_updates,
            )

        available[name] = EvaluationMethod(
            name=name,
            family=(
                "learned_on_policy"
                if checkpoint_algorithm == "hybrid_mappo"
                else "learned_off_policy"
            ),
            factory=factory,
            adaptation_updates=(
                0
                if checkpoint_algorithm == "hybrid_mappo"
                else args.adaptation_updates
            ),
            offline_training_seconds=training_times.get(name),
            checkpoint=checkpoint,
            notes=(
                "on-policy comparison; no online adaptation"
                if checkpoint_algorithm == "hybrid_mappo"
                else ""
            ),
        )
        requested_names.add(name)
    for name, target in _assignments(args.controller_factory, str).items():
        if ":" not in target:
            raise SystemExit(
                f"controller factory must be MODULE:CALLABLE, received {target!r}"
            )
        module_name, attribute = target.split(":", 1)
        external_factory = getattr(import_module(module_name), attribute)
        available[name] = EvaluationMethod(
            name=name,
            family="external_common_interface",
            factory=external_factory,
            adaptation_updates=args.adaptation_updates,
            offline_training_seconds=training_times.get(name),
            notes=f"factory={target}",
        )
        requested_names.add(name)
    unknown = requested_names - set(available)
    if unknown:
        raise SystemExit(f"unregistered methods: {sorted(unknown)}")
    methods = tuple(available[name] for name in sorted(requested_names))
    missing = missing_required_methods(methods)
    if args.require_complete_roster and missing:
        raise SystemExit(f"complete comparison requested; missing methods: {missing}")

    command = " ".join([sys.executable, *sys.argv])
    writer = EvidenceWriter(args.output, manifest, methods, command=command)
    stresses = tuple(value for value in args.stresses.split(",") if value) or None
    scenarios = manifest.select(split=args.split, stresses=stresses)
    runs = []
    for spec in scenarios:
        scenario = instantiate_scenario(spec)
        writer.write_scenario(scenario)
        for method in methods:
            run = evaluate_scenario(scenario, method)
            writer.write_run(run)
            runs.append(run)
            print(
                f"{run.scenario_id} {run.method}: "
                f"completion={run.metrics.completion_rate:.3f}, "
                f"deadline={run.metrics.deadline_success_rate:.3f}, "
                f"censored={run.metrics.unfinished_censored_tasks}"
            )
    writer.write_aggregate(runs, reference_method=args.reference_method)
    if not args.no_plots:
        generate_figures(runs, args.output / "figures")
        # Refresh hashes after plotting.
        writer.write_aggregate(runs, reference_method=args.reference_method)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "runs": len(runs),
                "missing_required_methods": missing,
                "comparison_complete": not missing,
            },
            indent=2,
        )
    )
    return 0


def _assignments(values, converter):
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected NAME=VALUE, received {value!r}")
        name, raw = value.split("=", 1)
        result[name] = converter(raw)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
