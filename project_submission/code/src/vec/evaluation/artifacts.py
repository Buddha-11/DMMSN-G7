"""Auditable evidence bundles: configs, traces, tables, and content hashes."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import sys
from typing import Iterable

from .runner import (
    REQUIRED_COMPARISON_METHODS,
    EvaluationMethod,
    EvaluationRun,
    missing_required_methods,
)
from .scenarios import ScenarioInstance, ScenarioManifest
from .statistics import confidence_interval, paired_comparison


CORE_METRICS = {
    "deadline_success_rate": True,
    "completion_rate": True,
    "throughput_tasks_per_s": True,
    "final_backlog_cycles": False,
    "backlog_growth_cycles_per_slot": False,
    "cumulative_reward": True,
    "controller_mean_seconds": False,
}

ENERGY_COMPONENTS = (
    "vehicle_compute",
    "vehicle_uplink",
    "rsu_compute",
    "rsu_reception",
    "uav_propulsion",
    "uav_computation",
    "uav_reception",
    "uav_backhaul",
    "uav_control",
    "uav_total",
    "cloud_compute",
    "cloud_reception",
    "system_total",
)


class EvidenceWriter:
    def __init__(
        self,
        output_directory: str | Path,
        manifest: ScenarioManifest,
        methods: tuple[EvaluationMethod, ...],
        *,
        command: str,
    ) -> None:
        self.root = Path(output_directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.methods = methods
        self.command = command
        manifest.save(self.root / "scenario_manifest.json")
        self._write_json(
            self.root / "method_roster.json",
            {
                "required_methods": list(REQUIRED_COMPARISON_METHODS),
                "registered_methods": [
                    {
                        "name": method.name,
                        "family": method.family,
                        "adaptation_updates": method.adaptation_updates,
                        "offline_training_seconds": method.offline_training_seconds,
                        "checkpoint": method.checkpoint,
                        "notes": method.notes,
                    }
                    for method in methods
                ],
                "missing_required_methods": list(missing_required_methods(methods)),
                "comparison_complete": not missing_required_methods(methods),
            },
        )
        self._write_json(
            self.root / "runtime.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: _package_version(name)
                    for name in ("numpy", "torch", "matplotlib")
                },
            },
        )
        self._write_json(
            self.root / "metric_protocol.json",
            {
                "confidence_intervals": (
                    "two-sided 95% Student-t intervals over independent "
                    "scenario seeds; normal approximation above 30 seeds"
                ),
                "paired_comparisons": (
                    "within-scenario differences on identical exogenous-trace "
                    "digests; unpaired keys are rejected"
                ),
                "deadline_success_rate": (
                    "tasks completed within their relative deadline divided by "
                    "all admitted tasks, including censored tasks in denominator"
                ),
                "response_time": "completed tasks only; censor count is mandatory",
                "backlog_growth": (
                    "ordinary least-squares slope over the offered-load horizon, "
                    "excluding drain slots"
                ),
                "drain_policy": (
                    "continue without new arrivals up to the configured drain "
                    "budget; otherwise report every unfinished task as censored"
                ),
                "recovery_time": (
                    "first post-change rolling window whose mean backlog is no "
                    "more than 110% of the pre-change rolling-window mean"
                ),
            },
        )
        (self.root / "reproduce.txt").write_text(command + "\n", encoding="utf-8")

    def write_scenario(self, scenario: ScenarioInstance) -> None:
        directory = self.root / "scenarios" / scenario.spec.scenario_id
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            directory / "configuration.json",
            {
                "spec": asdict(scenario.spec),
                "environment": asdict(scenario.config),
                "trace_digest": scenario.trace_digest,
                "offered_cycles": scenario.offered_cycles,
                "nominal_service_cycles_per_slot": (
                    scenario.nominal_service_cycles_per_slot
                ),
                "realized_offered_load_ratio": (
                    scenario.realized_offered_load_ratio
                ),
            },
        )
        self._write_json(
            directory / "exogenous_trace.json",
            {
                "arrivals": [asdict(item) for item in scenario.arrivals],
                "harvests": [asdict(item) for item in scenario.harvests],
                "mobility_source": scenario.spec.mobility_trace or "synthetic",
                "mobility_vehicle_ids": list(
                    scenario.spec.mobility_vehicle_ids
                ),
                "mobility_frames": scenario.mobility_frames,
            },
        )

    def write_run(self, run: EvaluationRun) -> None:
        directory = self.root / "runs" / run.scenario_id / run.method
        directory.mkdir(parents=True, exist_ok=True)
        payload = asdict(run)
        payload.pop("traces")
        payload.pop("tasks")
        self._write_json(directory / "summary.json", payload)
        with (directory / "slot_trace.jsonl").open("w", encoding="utf-8") as stream:
            for trace in run.traces:
                stream.write(json.dumps(asdict(trace), sort_keys=True) + "\n")
        task_fields = tuple(asdict(run.tasks[0])) if run.tasks else (
            "task_id",
            "vehicle_id",
            "arrival_time_s",
            "deadline_s",
            "completed",
            "response_time_s",
            "completed_within_deadline",
            "deadline_missed",
            "censor_age_s",
            "destination",
        )
        with (directory / "task_outcomes.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=task_fields)
            writer.writeheader()
            writer.writerows(asdict(task) for task in run.tasks)

    def write_aggregate(
        self,
        runs: Iterable[EvaluationRun],
        *,
        reference_method: str = "queue_energy_greedy",
    ) -> None:
        runs = tuple(runs)
        self._write_summary_table(runs)
        self._write_confidence_table(runs)
        self._write_paired_table(runs, reference_method)
        self._write_json(
            self.root / "evaluation_status.json",
            {
                "run_count": len(runs),
                "scenario_count": len({run.scenario_id for run in runs}),
                "methods_run": sorted({run.method for run in runs}),
                "trace_pairing_valid": _trace_pairing_valid(runs),
                "missing_required_methods": list(
                    missing_required_methods(self.methods)
                ),
                "comparison_complete": not missing_required_methods(self.methods),
                "limitations": (
                    []
                    if not missing_required_methods(self.methods)
                    else [
                        "Missing methods are not silently replaced; no complete "
                        "algorithm ranking may be claimed from this bundle."
                    ]
                ),
                "reproducible_command": self.command,
            },
        )
        self._write_hash_manifest()

    def _write_summary_table(self, runs: tuple[EvaluationRun, ...]) -> None:
        fields = (
            "scenario_id",
            "split",
            "stress",
            "seed",
            "method",
            "trace_digest",
            *CORE_METRICS,
            "response_time_mean_s",
            "response_time_p95_s",
            "response_time_p99_s",
            "unfinished_censored_tasks",
            *(f"energy_{name}_j" for name in ENERGY_COMPONENTS),
            "minimum_battery_charge_j",
            "final_virtual_energy_deficit_j",
            "peak_virtual_energy_deficit_j",
            "harvest_overflow_j",
            "landing_uav_slots",
            "unavailable_uav_slots",
            "repair_rate",
            "controller_p95_seconds",
            "physical_feasibility_failures",
            "battery_violations",
            "separation_violations",
            "online_updates",
            "online_update_seconds",
            "offline_training_seconds",
            "recovery_time_slots",
        )
        with (self.root / "run_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for run in runs:
                metrics = run.metrics
                writer.writerow(
                    {
                        "scenario_id": run.scenario_id,
                        "split": run.split,
                        "stress": run.stress,
                        "seed": run.seed,
                        "method": run.method,
                        "trace_digest": run.trace_digest,
                        **{
                            name: getattr(metrics, name) for name in CORE_METRICS
                        },
                        "response_time_mean_s": metrics.response_time_mean_s,
                        "response_time_p95_s": metrics.response_time_p95_s,
                        "response_time_p99_s": metrics.response_time_p99_s,
                        "unfinished_censored_tasks": (
                            metrics.unfinished_censored_tasks
                        ),
                        **{
                            f"energy_{name}_j": metrics.energy_j.get(name, 0.0)
                            for name in ENERGY_COMPONENTS
                        },
                        "minimum_battery_charge_j": min(
                            metrics.minimum_battery_charge_j, default=0.0
                        ),
                        "final_virtual_energy_deficit_j": max(
                            metrics.final_virtual_energy_deficit_j, default=0.0
                        ),
                        "peak_virtual_energy_deficit_j": max(
                            metrics.peak_virtual_energy_deficit_j, default=0.0
                        ),
                        "harvest_overflow_j": metrics.harvest_overflow_j,
                        "landing_uav_slots": metrics.landing_uav_slots,
                        "unavailable_uav_slots": metrics.unavailable_uav_slots,
                        "repair_rate": metrics.repair_rate,
                        "controller_p95_seconds": metrics.controller_p95_seconds,
                        "physical_feasibility_failures": (
                            metrics.physical_feasibility_failures
                        ),
                        "battery_violations": metrics.battery_violations,
                        "separation_violations": metrics.separation_violations,
                        "online_updates": metrics.online_updates,
                        "online_update_seconds": metrics.online_update_seconds,
                        "offline_training_seconds": metrics.offline_training_seconds,
                        "recovery_time_slots": metrics.recovery_time_slots,
                    }
                )

    def _write_confidence_table(self, runs: tuple[EvaluationRun, ...]) -> None:
        rows: list[dict] = []
        groups = sorted({(run.method, run.stress) for run in runs})
        for method, stress in groups:
            selected = [
                run for run in runs if run.method == method and run.stress == stress
            ]
            for metric in CORE_METRICS:
                interval = confidence_interval(
                    [float(getattr(run.metrics, metric)) for run in selected]
                )
                rows.append(
                    {
                        "method": method,
                        "stress": stress,
                        "metric": metric,
                        **asdict(interval),
                    }
                )
        _write_csv(self.root / "confidence_intervals.csv", rows)

    def _write_paired_table(
        self,
        runs: tuple[EvaluationRun, ...],
        reference_method: str,
    ) -> None:
        rows: list[dict] = []
        methods = sorted({run.method for run in runs if run.method != reference_method})
        stresses = sorted({run.stress for run in runs})
        for method in methods:
            for stress in stresses:
                candidates = {
                    (run.scenario_id, run.seed): run
                    for run in runs
                    if run.method == method and run.stress == stress
                }
                reference = {
                    (run.scenario_id, run.seed): run
                    for run in runs
                    if run.method == reference_method and run.stress == stress
                }
                if not candidates or not reference:
                    continue
                for metric, higher_is_better in CORE_METRICS.items():
                    comparison = paired_comparison(
                        {
                            key: float(getattr(run.metrics, metric))
                            for key, run in candidates.items()
                        },
                        {
                            key: float(getattr(run.metrics, metric))
                            for key, run in reference.items()
                        },
                        method=method,
                        reference_method=reference_method,
                        metric=metric,
                        higher_is_better=higher_is_better,
                    )
                    rows.append({"stress": stress, **asdict(comparison)})
        _write_csv(self.root / "paired_comparisons.csv", rows)

    def _write_hash_manifest(self) -> None:
        hashes = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                hashes[path.relative_to(self.root).as_posix()] = sha256(
                    path.read_bytes()
                ).hexdigest()
        self._write_json(
            self.root / "artifact_manifest.json",
            {"hash_algorithm": "sha256", "files": hashes},
        )

    @staticmethod
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


def _trace_pairing_valid(runs: tuple[EvaluationRun, ...]) -> bool:
    by_scenario: dict[str, set[str]] = {}
    for run in runs:
        by_scenario.setdefault(run.scenario_id, set()).add(run.trace_digest)
    return all(len(digests) == 1 for digests in by_scenario.values())


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
