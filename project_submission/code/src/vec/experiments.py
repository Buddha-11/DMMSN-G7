"""Multi-seed study orchestration and learner-seed statistical aggregation."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from itertools import combinations
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

from .evaluation.artifacts import CORE_METRICS, ENERGY_COMPONENTS
from .evaluation.runner import REQUIRED_COMPARISON_METHODS
from .evaluation.scenarios import ScenarioManifest, create_standard_manifest
from .evaluation.study_plots import generate_study_figures
from .evaluation.statistics import confidence_interval, paired_comparison


STUDY_SCHEMA_VERSION = "vec-study/v1"
LEARNED_METHODS = (
    "hybrid_maddpg",
    "domain_randomized_maddpg",
    "reptile_maddpg",
    "reptile_no_queue_weight",
    "hybrid_mappo",
)
DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS = (
    "domain_randomized_maddpg",
    "domain_randomized_no_queue_weight",
)
SUPPORTED_LEARNED_METHODS = tuple(
    dict.fromkeys((*LEARNED_METHODS, *DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS))
)
BASELINE_METHODS = (
    "local_only",
    "nearest_feasible_edge",
    "queue_energy_greedy",
    "genetic_drift",
)
MIN_RESEARCH_LEARNER_SEEDS = 5
SAFETY_METRICS = {
    "physical_feasibility_failures": False,
    "battery_violations": False,
    "separation_violations": False,
}
STUDY_METRICS = {
    **CORE_METRICS,
    "unfinished_censored_tasks": False,
    **{f"energy_{name}_j": False for name in ENERGY_COMPONENTS},
    "minimum_battery_charge_j": True,
    "final_virtual_energy_deficit_j": False,
    "peak_virtual_energy_deficit_j": False,
    "harvest_overflow_j": False,
    "landing_uav_slots": False,
    "unavailable_uav_slots": False,
    "repair_rate": False,
    "controller_p95_seconds": False,
    "online_updates": False,
    **SAFETY_METRICS,
}
OPTIONAL_STUDY_METRICS = {
    "response_time_mean_s": False,
    "response_time_p95_s": False,
    "response_time_p99_s": False,
    "online_update_seconds": False,
    "offline_training_seconds": False,
    "recovery_time_slots": False,
}


@dataclass(frozen=True)
class StudyConfig:
    learner_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    train_seeds: tuple[int, ...] = tuple(range(10))
    validation_seeds: tuple[int, ...] = tuple(range(100, 105))
    test_seeds: tuple[int, ...] = tuple(range(200, 210))
    epochs: int = 50
    scenarios_per_epoch: int = 7
    horizon_slots: int = 200
    drain_slots: int = 100
    genetic_budget: int = 64
    device: str = "cpu"
    smoke: bool = False
    schema_version: str = STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_SCHEMA_VERSION:
            raise ValueError("unsupported study schema")
        for name in (
            "learner_seeds",
            "train_seeds",
            "validation_seeds",
            "test_seeds",
        ):
            values = tuple(int(value) for value in getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique seeds")
            object.__setattr__(self, name, values)
        scenario_splits = (
            set(self.train_seeds),
            set(self.validation_seeds),
            set(self.test_seeds),
        )
        if any(
            first & second
            for index, first in enumerate(scenario_splits)
            for second in scenario_splits[index + 1 :]
        ):
            raise ValueError("train, validation, and test scenario seeds must be disjoint")
        if any(
            value <= 0
            for value in (
                self.epochs,
                self.scenarios_per_epoch,
                self.horizon_slots,
                self.genetic_budget,
            )
        ) or self.drain_slots < 0:
            raise ValueError("study counts and horizons must be positive")
        if not self.device:
            raise ValueError("device cannot be empty")


@dataclass(frozen=True)
class StudyJob:
    method: str
    learner_seed: int
    output_directory: str
    command: tuple[str, ...]


class VecStudyRunner:
    """Run every learned method with shared splits and auditable resume state."""

    def __init__(
        self,
        config: StudyConfig,
        output_directory: str | Path,
        *,
        project_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        learned_methods: Iterable[str] | None = None,
        baseline_methods: Iterable[str] | None = None,
        require_complete_roster: bool = True,
    ) -> None:
        self.config = config
        self.output = Path(output_directory).resolve()
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.python = str(Path(python_executable or sys.executable).resolve())
        self.manifest_path = self.output / "scenario_manifest.json"
        self.learned_methods = tuple(learned_methods or LEARNED_METHODS)
        self.baseline_methods = tuple(
            BASELINE_METHODS if baseline_methods is None else baseline_methods
        )
        unsupported = set(self.learned_methods) - set(SUPPORTED_LEARNED_METHODS)
        if unsupported:
            raise ValueError(f"unsupported learned methods: {sorted(unsupported)}")
        if not self.learned_methods or len(self.learned_methods) != len(
            set(self.learned_methods)
        ):
            raise ValueError("learned_methods must contain unique method names")
        if len(self.baseline_methods) != len(set(self.baseline_methods)):
            raise ValueError("baseline_methods must contain unique method names")
        self.require_complete_roster = bool(require_complete_roster)
        self._workers = 1

    def prepare(self) -> tuple[StudyJob, ...]:
        self.output.mkdir(parents=True, exist_ok=True)
        expected_manifest = self._manifest()
        config_path = self.output / "study_config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != _jsonable(asdict(self.config)):
                raise ValueError("study output already contains a different configuration")
        else:
            _write_json(config_path, asdict(self.config))
        if self.manifest_path.exists():
            if ScenarioManifest.load(self.manifest_path) != expected_manifest:
                raise ValueError("study output already contains a different scenario manifest")
        else:
            expected_manifest.save(self.manifest_path)
        jobs = tuple(
            self._training_job(method, seed)
            for seed in self.config.learner_seeds
            for method in self.learned_methods
        )
        _write_json(
            self.output / "study_plan.json",
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "learned_methods": self.learned_methods,
                "baseline_methods": self.baseline_methods,
                "required_comparison_methods": (
                    REQUIRED_COMPARISON_METHODS
                    if self.require_complete_roster
                    else self.learned_methods
                ),
                "require_complete_roster": self.require_complete_roster,
                "training_jobs": [asdict(job) for job in jobs],
                "evaluation_outputs": [
                    str(self._evaluation_output(seed))
                    for seed in self.config.learner_seeds
                ],
                "statistical_unit": (
                    "independent learner seed after averaging paired test scenarios"
                ),
            },
        )
        self._write_status(stage="planned")
        return jobs

    def run(
        self,
        *,
        dry_run: bool = False,
        skip_evaluation: bool = False,
        plots: bool = True,
        workers: int = 1,
    ) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self._workers = int(workers)
        jobs = self.prepare()
        if dry_run:
            return
        pending_jobs = [
            job
            for job in jobs
            if not (
                (Path(job.output_directory) / "best.pt").is_file()
                and (Path(job.output_directory) / "training_summary.json").is_file()
            )
        ]
        if self._workers == 1:
            for job in pending_jobs:
                self._run_training_job(job)
                self._write_status(stage="training")
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as executor:
                futures = {
                    executor.submit(self._run_training_job, job): job
                    for job in pending_jobs
                }
                for future in as_completed(futures):
                    future.result()
                    self._write_status(stage="training")
        if skip_evaluation:
            self._write_status(stage="training_complete")
            return
        pending_evaluations = [
            seed
            for seed in self.config.learner_seeds
            if not self._evaluation_complete(seed)
        ]
        if self._workers == 1:
            for seed in pending_evaluations:
                self._run_evaluation(seed, plots=plots)
                self._write_status(stage="evaluation")
        else:
            with ThreadPoolExecutor(max_workers=self._workers) as executor:
                futures = {
                    executor.submit(self._run_evaluation, seed, plots=plots): seed
                    for seed in pending_evaluations
                }
                for future in as_completed(futures):
                    future.result()
                    self._write_status(stage="evaluation")
        aggregate_study_evaluations(
            self.output,
            self.config.learner_seeds,
            reference_method=(
                "queue_energy_greedy"
                if "queue_energy_greedy" in self.baseline_methods
                else self.learned_methods[0]
            ),
            smoke=self.config.smoke,
            required_methods=self.learned_methods,
            require_complete_roster=self.require_complete_roster,
        )
        if plots:
            generate_study_figures(self.output)
        self._write_status(stage="complete")
        self._write_artifact_manifest()

    def _manifest(self) -> ScenarioManifest:
        if self.config.smoke:
            seeds = {"train": (0,), "validation": (100,), "test": (200,)}
            horizon, drain = 4, 2
            identifier = "vec-study-smoke-v1"
        else:
            seeds = {
                "train": self.config.train_seeds,
                "validation": self.config.validation_seeds,
                "test": self.config.test_seeds,
            }
            horizon, drain = self.config.horizon_slots, self.config.drain_slots
            identifier = "vec-research-study-v1"
        return create_standard_manifest(
            seeds,
            horizon_slots=horizon,
            drain_slots=drain,
            manifest_id=identifier,
        )

    def _training_job(self, method: str, seed: int) -> StudyJob:
        output = self.output / "training" / method / f"seed-{seed:03d}"
        if method == "hybrid_mappo":
            script = self.project_root / "scripts" / "train_vec_mappo.py"
            method_args: list[str] = []
        else:
            script = self.project_root / "scripts" / "train_vec.py"
            modes = {
                "hybrid_maddpg": "maddpg",
                "domain_randomized_maddpg": "domain_randomized",
                "domain_randomized_no_queue_weight": "domain_randomized",
                "reptile_maddpg": "reptile",
                "reptile_no_queue_weight": "reptile",
            }
            method_args = ["--mode", modes[method]]
            if method in {
                "domain_randomized_no_queue_weight",
                "reptile_no_queue_weight",
            }:
                method_args.append("--no-queue-weight")
        command = [
            self.python,
            str(script),
            "--output",
            str(output),
            "--manifest",
            str(self.manifest_path),
            "--epochs",
            str(self.config.epochs),
            "--scenarios-per-epoch",
            str(self.config.scenarios_per_epoch),
            "--seed",
            str(seed),
            "--device",
            self.config.device,
            *method_args,
        ]
        if self.config.smoke:
            command.append("--smoke")
        return StudyJob(method, seed, str(output), tuple(command))

    def _evaluation_command(self, seed: int, *, plots: bool) -> list[str]:
        output = self._evaluation_output(seed)
        command = [
            self.python,
            str(self.project_root / "scripts" / "evaluate_vec.py"),
            "--output",
            str(output),
            "--manifest",
            str(self.manifest_path),
            "--split",
            "test",
            "--methods",
            ",".join(self.baseline_methods),
            "--genetic-budget",
            str(self.config.genetic_budget),
            "--reference-method",
            (
                "queue_energy_greedy"
                if "queue_energy_greedy" in self.baseline_methods
                else self.learned_methods[0]
            ),
        ]
        if self.require_complete_roster:
            command.append("--require-complete-roster")
        if not plots:
            command.append("--no-plots")
        for method in self.learned_methods:
            training = self.output / "training" / method / f"seed-{seed:03d}"
            command.extend(["--checkpoint", f"{method}={training / 'best.pt'}"])
            summary = json.loads(
                (training / "training_summary.json").read_text(encoding="utf-8")
            )
            command.extend(
                [
                    "--offline-training-seconds",
                    f"{method}={float(summary['offline_training_seconds'])}",
                ]
            )
        return command

    def _evaluation_output(self, seed: int) -> Path:
        return self.output / "evaluation" / f"seed-{seed:03d}"

    def _run_training_job(self, job: StudyJob) -> None:
        root = Path(job.output_directory)
        checkpoint = root / "best.pt"
        command = list(job.command)
        if (root / "training_state.pt").is_file():
            command.append("--resume")
        self._run_command(
            command,
            self.output / "logs" / f"train-{job.method}-s{job.learner_seed}.log",
        )
        if not checkpoint.is_file():
            raise RuntimeError(
                f"{job.method} seed {job.learner_seed} produced no safety-eligible best.pt"
            )

    def _evaluation_complete(self, seed: int) -> bool:
        status_path = self._evaluation_output(seed) / "evaluation_status.json"
        if not status_path.is_file():
            return False
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return bool(
            status.get("trace_pairing_valid")
            and (
                status.get("comparison_complete")
                if self.require_complete_roster
                else set(self.learned_methods).issubset(status.get("methods_run", ()))
            )
        )

    def _run_evaluation(self, seed: int, *, plots: bool) -> None:
        self._run_command(
            self._evaluation_command(seed, plots=plots),
            self.output / "logs" / f"evaluate-s{seed}.log",
        )
        if not self._evaluation_complete(seed):
            raise RuntimeError(f"evaluation seed {seed} is incomplete or unpaired")

    def _run_command(self, command: Iterable[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                tuple(command),
                cwd=self.project_root,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                env=self._subprocess_environment(),
            )
        if completed.returncode:
            self._write_status(
                stage="failed",
                failure={
                    "command": list(command),
                    "returncode": completed.returncode,
                    "log": str(log_path),
                },
            )
            raise RuntimeError(
                f"study command failed with exit code {completed.returncode}; see {log_path}"
            )

    def _subprocess_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        matplotlib_config = self.output / ".matplotlib"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        environment["MPLCONFIGDIR"] = str(matplotlib_config)
        environment["PYTHONUNBUFFERED"] = "1"
        if self._workers > 1:
            threads = max(1, (os.cpu_count() or 1) // self._workers)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = str(threads)
        return environment

    def _write_status(self, *, stage: str, failure: dict | None = None) -> None:
        completed_training = []
        for seed in self.config.learner_seeds:
            for method in self.learned_methods:
                root = self.output / "training" / method / f"seed-{seed:03d}"
                if (root / "best.pt").is_file() and (
                    root / "training_summary.json"
                ).is_file():
                    completed_training.append({"method": method, "learner_seed": seed})
        completed_evaluations = []
        for seed in self.config.learner_seeds:
            path = self._evaluation_output(seed) / "evaluation_status.json"
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("trace_pairing_valid") and (
                    payload.get("comparison_complete")
                    if self.require_complete_roster
                    else set(self.learned_methods).issubset(
                        payload.get("methods_run", ())
                    )
                ):
                    completed_evaluations.append(seed)
        _write_json(
            self.output / "study_status.json",
            {
                "schema_version": STUDY_SCHEMA_VERSION,
                "stage": stage,
                "smoke": self.config.smoke,
                "training_jobs_total": len(self.config.learner_seeds)
                * len(self.learned_methods),
                "training_jobs_completed": completed_training,
                "evaluation_seeds_completed": completed_evaluations,
                "failure": failure,
            },
        )

    def _write_artifact_manifest(self) -> None:
        result_names = (
            "study_config.json",
            "study_plan.json",
            "study_status.json",
            "scenario_manifest.json",
            "study_run_metrics.csv",
            "learner_seed_metrics.csv",
            "learner_seed_confidence_intervals.csv",
            "learner_seed_paired_comparisons.csv",
            "learner_seed_pairwise_comparisons.csv",
            "study_summary.json",
        )
        result_files = {
            name: sha256((self.output / name).read_bytes()).hexdigest()
            for name in result_names
            if (self.output / name).is_file()
        }
        checkpoints = {
            path.relative_to(self.output).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.output.glob("training/*/seed-*/best.pt"))
        }
        evaluation_manifests = {
            path.relative_to(self.output).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                self.output.glob("evaluation/seed-*/artifact_manifest.json")
            )
        }
        figure_files = {
            path.relative_to(self.output).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted((self.output / "figures").glob("*.png"))
        }
        source_candidates = [
            *sorted((self.project_root / "src" / "vec").rglob("*.py")),
            self.project_root / "scripts" / "train_vec.py",
            self.project_root / "scripts" / "train_vec_mappo.py",
            self.project_root / "scripts" / "evaluate_vec.py",
            self.project_root / "scripts" / "run_vec_study.py",
            self.project_root
            / "scripts"
            / "run_domain_randomized_queue_ablation.py",
            self.project_root / "requirements.txt",
        ]
        source_files = {
            path.relative_to(self.project_root).as_posix(): sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_candidates
            if path.is_file()
        }
        manifest = ScenarioManifest.load(self.manifest_path)
        external_mobility = sorted(
            {
                scenario.mobility_trace
                for scenario in manifest.scenarios
                if scenario.mobility_trace is not None
            }
        )
        dependencies = {
            name: _installed_version(name)
            for name in ("numpy", "torch", "matplotlib", "imageio", "imageio-ffmpeg")
        }
        _write_json(
            self.output / "study_artifact_manifest.json",
            {
                "schema_version": "vec-study-artifacts/v2",
                "study_schema_version": STUDY_SCHEMA_VERSION,
                "hash_algorithm": "sha256",
                "result_files": result_files,
                "selected_checkpoints": {
                    "selection_rule": (
                        "highest validation mean reward per slot among checkpoints "
                        "with zero validation physical violations; test split sealed"
                    ),
                    "expected_count": len(self.config.learner_seeds)
                    * len(self.learned_methods),
                    "files": checkpoints,
                },
                "evaluation_bundle_manifests": evaluation_manifests,
                "figures": figure_files,
                "source_files": source_files,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "python_executable": self.python,
                    "dependencies": dependencies,
                },
                "seeds": {
                    "learner": self.config.learner_seeds,
                    "train_scenarios": self.config.train_seeds,
                    "validation_scenarios": self.config.validation_seeds,
                    "sealed_test_scenarios": self.config.test_seeds,
                },
                "datasets": {
                    "external_datasets_used": bool(external_mobility),
                    "external_mobility_traces": external_mobility,
                    "generated_sources": [
                        {
                            "name": "synthetic road-constrained vehicle mobility",
                            "implementation": "src/vec/mobility.py",
                            "license": (
                                "not applicable: procedurally generated by project code; "
                                "no external dataset is redistributed"
                            ),
                        },
                        {
                            "name": "synthetic task-arrival and energy-harvest traces",
                            "implementation": "src/vec/evaluation/scenarios.py",
                            "license": (
                                "not applicable: procedurally generated by project code; "
                                "no external dataset is redistributed"
                            ),
                        },
                    ],
                    "note": (
                        "The frozen research-v1 study uses synthetic mobility and workload "
                        "traces. SUMO/Bologna support is implemented but is not evidence in "
                        "this result bundle."
                    ),
                },
            },
        )


def aggregate_study_evaluations(
    study_directory: str | Path,
    learner_seeds: Iterable[int],
    *,
    reference_method: str = "queue_energy_greedy",
    smoke: bool = False,
    required_methods: Iterable[str] | None = None,
    require_complete_roster: bool = True,
) -> None:
    """Aggregate paired test scenarios with learner seed as the sampling unit."""

    root = Path(study_directory)
    raw_rows: list[dict[str, str | int]] = []
    seeds = tuple(int(seed) for seed in learner_seeds)
    requested_methods = tuple(required_methods or REQUIRED_COMPARISON_METHODS)
    for learner_seed in seeds:
        evaluation = root / "evaluation" / f"seed-{learner_seed:03d}"
        status = json.loads(
            (evaluation / "evaluation_status.json").read_text(encoding="utf-8")
        )
        status_methods = status.get("methods_run")
        requested_present = (
            set(requested_methods).issubset(status_methods)
            if status_methods is not None
            else bool(status.get("comparison_complete"))
        )
        if (
            not status.get("trace_pairing_valid")
            or not requested_present
            or (require_complete_roster and not status.get("comparison_complete"))
        ):
            raise ValueError(f"evaluation seed {learner_seed} is incomplete or unpaired")
        with (evaluation / "run_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                raw_rows.append({"learner_seed": learner_seed, **row})
    _write_csv(root / "study_run_metrics.csv", raw_rows)

    seed_rows: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in raw_rows})
    stresses = sorted({str(row["stress"]) for row in raw_rows})
    for learner_seed in seeds:
        for method in methods:
            selected_method = [
                row
                for row in raw_rows
                if row["learner_seed"] == learner_seed and row["method"] == method
            ]
            for stress in (*stresses, "__all__"):
                selected = (
                    selected_method
                    if stress == "__all__"
                    else [row for row in selected_method if row["stress"] == stress]
                )
                if not selected:
                    continue
                seed_rows.append(
                    {
                        "learner_seed": learner_seed,
                        "method": method,
                        "stress": stress,
                        "scenario_count": len(selected),
                        **{
                            metric: sum(float(row[metric]) for row in selected)
                            / len(selected)
                            for metric in STUDY_METRICS
                        },
                        **{
                            metric: _mean_present(selected, metric)
                            for metric in OPTIONAL_STUDY_METRICS
                        },
                    }
                )
    _write_csv(root / "learner_seed_metrics.csv", seed_rows)

    interval_rows: list[dict[str, object]] = []
    for method in methods:
        for stress in (*stresses, "__all__"):
            selected = [
                row
                for row in seed_rows
                if row["method"] == method and row["stress"] == stress
            ]
            if not selected:
                continue
            for metric in (*STUDY_METRICS, *OPTIONAL_STUDY_METRICS):
                values = [
                    float(row[metric])
                    for row in selected
                    if _has_numeric_value(row.get(metric))
                ]
                if not values:
                    continue
                interval_rows.append(
                    {
                        "method": method,
                        "stress": stress,
                        "metric": metric,
                        **asdict(
                            confidence_interval(values)
                        ),
                    }
                )
    _write_csv(root / "learner_seed_confidence_intervals.csv", interval_rows)

    comparison_rows: list[dict[str, object]] = []
    for method in methods:
        if method == reference_method:
            continue
        for stress in (*stresses, "__all__"):
            candidates = {
                int(row["learner_seed"]): row
                for row in seed_rows
                if row["method"] == method and row["stress"] == stress
            }
            reference = {
                int(row["learner_seed"]): row
                for row in seed_rows
                if row["method"] == reference_method and row["stress"] == stress
            }
            if not candidates or not reference:
                continue
            for metric, higher_is_better in {
                **STUDY_METRICS,
                **OPTIONAL_STUDY_METRICS,
            }.items():
                common = sorted(set(candidates) & set(reference))
                common = [
                    seed
                    for seed in common
                    if _has_numeric_value(candidates[seed].get(metric))
                    and _has_numeric_value(reference[seed].get(metric))
                ]
                if not common:
                    continue
                comparison_rows.append(
                    {
                        "stress": stress,
                        **asdict(
                            paired_comparison(
                                {
                                    seed: float(candidates[seed][metric])
                                    for seed in common
                                },
                                {
                                    seed: float(reference[seed][metric])
                                    for seed in common
                                },
                                method=method,
                                reference_method=reference_method,
                                metric=metric,
                                higher_is_better=higher_is_better,
                            )
                        ),
                    }
                )
    _write_csv(root / "learner_seed_paired_comparisons.csv", comparison_rows)
    pairwise_rows: list[dict[str, object]] = []
    for method, reference in combinations(methods, 2):
        for stress in (*stresses, "__all__"):
            candidate_rows = {
                int(row["learner_seed"]): row
                for row in seed_rows
                if row["method"] == method and row["stress"] == stress
            }
            reference_rows = {
                int(row["learner_seed"]): row
                for row in seed_rows
                if row["method"] == reference and row["stress"] == stress
            }
            for metric, higher_is_better in {
                **STUDY_METRICS,
                **OPTIONAL_STUDY_METRICS,
            }.items():
                common = sorted(set(candidate_rows) & set(reference_rows))
                common = [
                    seed
                    for seed in common
                    if _has_numeric_value(candidate_rows[seed].get(metric))
                    and _has_numeric_value(reference_rows[seed].get(metric))
                ]
                if not common:
                    continue
                pairwise_rows.append(
                    {
                        "stress": stress,
                        **asdict(
                            paired_comparison(
                                {
                                    seed: float(candidate_rows[seed][metric])
                                    for seed in common
                                },
                                {
                                    seed: float(reference_rows[seed][metric])
                                    for seed in common
                                },
                                method=method,
                                reference_method=reference,
                                metric=metric,
                                higher_is_better=higher_is_better,
                            )
                        ),
                    }
                )
    _write_csv(root / "learner_seed_pairwise_comparisons.csv", pairwise_rows)
    all_methods_present = set(REQUIRED_COMPARISON_METHODS).issubset(methods)
    requested_methods_present = set(requested_methods).issubset(methods)
    safety_valid = all(
        sum(float(row[metric]) for row in raw_rows) == 0.0
        for metric in SAFETY_METRICS
    )
    protocol_scale_valid = _research_scale_valid(root, seeds, smoke=smoke)
    claim_ready = (
        protocol_scale_valid
        and requested_methods_present
        and (all_methods_present or not require_complete_roster)
        and safety_valid
    )
    _write_json(
        root / "study_summary.json",
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "learner_seed_count": len(seeds),
            "learner_seeds": seeds,
            "methods": methods,
            "test_run_count": len(raw_rows),
            "statistical_unit": (
                "independent learner seed after averaging paired test scenarios"
            ),
            "comparison_complete": all_methods_present,
            "requested_methods": requested_methods,
            "requested_comparison_complete": requested_methods_present,
            "require_complete_roster": require_complete_roster,
            "safety_valid": safety_valid,
            "protocol_scale_valid": protocol_scale_valid,
            "research_claim_ready": claim_ready,
            "limitations": [
                message
                for condition, message in (
                    (smoke, "Smoke configuration is an integration test, not evidence."),
                    (
                        len(seeds) < MIN_RESEARCH_LEARNER_SEEDS,
                        "Fewer than five learner seeds do not meet the preregistered study protocol.",
                    ),
                    (
                        not requested_methods_present,
                        "One or more requested methods are missing.",
                    ),
                    (
                        require_complete_roster and not all_methods_present,
                        "The complete benchmark roster is incomplete.",
                    ),
                    (
                        not safety_valid,
                        "One or more physical, battery, or separation violations were recorded.",
                    ),
                    (
                        not smoke and not protocol_scale_valid,
                        "The run does not meet the frozen minimum research-scale configuration.",
                    ),
                )
                if condition
            ],
            "readiness_scope": (
                "Protocol completeness only; convergence and substantive claims still "
                "require inspection of learning curves and effect estimates."
            ),
        },
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _has_numeric_value(value: object) -> bool:
    return value not in (None, "")


def _mean_present(rows: list[dict[str, object]], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if _has_numeric_value(row.get(metric))
    ]
    return sum(values) / len(values) if values else None


def _research_scale_valid(root: Path, seeds: tuple[int, ...], *, smoke: bool) -> bool:
    if smoke or len(seeds) < MIN_RESEARCH_LEARNER_SEEDS:
        return False
    config_path = root / "study_config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return (
        len(config.get("learner_seeds", ())) >= MIN_RESEARCH_LEARNER_SEEDS
        and set(map(int, config["learner_seeds"])) == set(seeds)
        and len(config.get("train_seeds", ())) >= 10
        and len(config.get("validation_seeds", ())) >= 5
        and len(config.get("test_seeds", ())) >= 10
        and int(config.get("epochs", 0)) >= 50
        and int(config.get("scenarios_per_epoch", 0)) >= 7
        and int(config.get("horizon_slots", 0)) >= 200
        and int(config.get("drain_slots", -1)) >= 100
        and int(config.get("genetic_budget", 0)) >= 64
    )


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not rows:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
