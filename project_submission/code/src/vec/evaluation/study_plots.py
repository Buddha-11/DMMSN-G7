"""Learner-seed-level figures for a completed multi-seed VEC study."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
from statistics import fmean

import numpy as np

from .statistics import confidence_interval


METHOD_ORDER = (
    "local_only",
    "nearest_feasible_edge",
    "queue_energy_greedy",
    "genetic_drift",
    "hybrid_maddpg",
    "domain_randomized_maddpg",
    "domain_randomized_no_queue_weight",
    "reptile_maddpg",
    "reptile_no_queue_weight",
    "hybrid_mappo",
)
STRESS_ORDER = (
    "low_load",
    "near_capacity",
    "overload",
    "zero_harvest",
    "harvest_outage",
    "backhaul_bottleneck",
    "traffic_surge",
)


def generate_study_figures(study_directory: str | Path) -> tuple[Path, ...]:
    """Generate figures whose uncertainty unit is the independent learner seed."""

    root = Path(study_directory)
    matplotlib_config = root / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib.pyplot as plt

    seed_metrics = _read_csv(root / "learner_seed_metrics.csv")
    if not seed_metrics:
        raise ValueError("learner_seed_metrics.csv is empty")
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = (
        output / "study_headline_metrics.png",
        output / "study_stress_heatmap.png",
        output / "study_learning_curves.png",
        output / "study_adaptation_curves.png",
    )
    _headline_plot(plt, seed_metrics, paths[0])
    _stress_heatmap(plt, seed_metrics, paths[1])
    _learning_curves(plt, root, paths[2])
    _adaptation_curves(plt, root, paths[3])
    return paths


def _headline_plot(plt, rows: list[dict[str, str]], path: Path) -> None:
    methods = _ordered({row["method"] for row in rows})
    selected = [row for row in rows if row["stress"] == "__all__"]
    fields = (
        ("deadline_success_rate", "Deadline success", 1.0),
        ("completion_rate", "Completion rate", 1.0),
        ("final_backlog_cycles", "Final backlog (Gcycles)", 1.0e9),
        ("energy_system_total_j", "System energy (kJ)", 1.0e3),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (metric, title, scale) in zip(axes.flat, fields, strict=True):
        means, errors = [], []
        for method in methods:
            values = [
                float(row[metric]) / scale
                for row in selected
                if row["method"] == method and row.get(metric) not in (None, "")
            ]
            interval = confidence_interval(values)
            means.append(interval.mean)
            errors.append(interval.upper - interval.mean)
        axis.bar(range(len(methods)), means, yerr=errors, capsize=3)
        axis.set_title(f"{title} (95% learner-seed CI)")
        axis.set_xticks(range(len(methods)), [_short(method) for method in methods])
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _stress_heatmap(plt, rows: list[dict[str, str]], path: Path) -> None:
    selected = [row for row in rows if row["stress"] != "__all__"]
    methods = _ordered({row["method"] for row in selected})
    available_stresses = {row["stress"] for row in selected}
    stresses = [stress for stress in STRESS_ORDER if stress in available_stresses]
    values = np.zeros((len(methods), len(stresses)), dtype=float)
    for method_index, method in enumerate(methods):
        for stress_index, stress in enumerate(stresses):
            samples = [
                float(row["completion_rate"])
                for row in selected
                if row["method"] == method and row["stress"] == stress
            ]
            values[method_index, stress_index] = fmean(samples)
    figure, axis = plt.subplots(
        figsize=(12, 6.5), constrained_layout=True
    )
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(
        range(len(stresses)),
        [stress.replace("_", " ").title() for stress in stresses],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(methods)), [_short(method) for method in methods])
    axis.set_title("Completion rate by stress (mean across learner seeds)")
    for row_index in range(len(methods)):
        for column_index in range(len(stresses)):
            value = values[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Completion rate")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _learning_curves(plt, root: Path, path: Path) -> None:
    histories: dict[str, dict[int, list[dict[str, object]]]] = defaultdict(dict)
    for history_path in sorted(root.glob("training/*/seed-*/history.jsonl")):
        method = history_path.parent.parent.name
        seed = int(history_path.parent.name.split("-")[-1])
        histories[method][seed] = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    methods = _ordered(histories)
    if not methods:
        figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
        axis.axis("off")
        axis.text(0.5, 0.5, "No training histories found", ha="center", va="center")
        figure.savefig(path, dpi=220)
        plt.close(figure)
        return
    columns = 2
    rows = (len(methods) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3.4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, method in zip(axes.flat, methods):
        for seed, records in sorted(histories[method].items()):
            epochs = [int(record["epoch"]) for record in records]
            scores = [
                float(record["validation_mean_reward_per_slot"])
                for record in records
            ]
            axis.plot(
                epochs,
                scores,
                marker="o",
                markersize=2.5,
                alpha=0.75,
                label=f"seed {seed}",
            )
        axis.set_title(_short(method))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation reward / slot")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    for axis in axes.flat[len(methods) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Validation learning curves (individual learner seeds; early stopping retained)"
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _adaptation_curves(plt, root: Path, path: Path) -> None:
    by_method_seed_offset: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for summary_path in sorted(
        root.glob(
            "evaluation/seed-*/runs/test-traffic_surge-*/**/summary.json"
        )
    ):
        learner_seed = int(summary_path.parents[3].name.split("-")[-1])
        method = summary_path.parent.name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        curve = payload["metrics"]["adaptation_curve_reward"]
        for offset, reward in curve.items():
            by_method_seed_offset[method][learner_seed][int(offset)].append(
                float(reward)
            )
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    if not by_method_seed_offset:
        axis.axis("off")
        axis.text(0.5, 0.5, "No traffic-surge adaptation traces found", ha="center")
        figure.savefig(path, dpi=220)
        plt.close(figure)
        return
    for method in _ordered(by_method_seed_offset):
        offsets = sorted(
            {
                offset
                for by_offset in by_method_seed_offset[method].values()
                for offset in by_offset
            }
        )
        means, lower, upper = [], [], []
        for offset in offsets:
            seed_values = []
            for by_offset in by_method_seed_offset[method].values():
                if offset in by_offset:
                    seed_values.append(fmean(by_offset[offset]))
            interval = confidence_interval(seed_values)
            means.append(interval.mean)
            lower.append(interval.lower)
            upper.append(interval.upper)
        axis.plot(offsets, means, marker="o", label=_short(method))
        axis.fill_between(offsets, lower, upper, alpha=0.12)
    axis.set_title("Traffic-surge adaptation (95% learner-seed CI)")
    axis.set_xlabel("Slots since declared change")
    axis.set_ylabel("Mean reward through offset")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=3)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _ordered(methods) -> list[str]:
    available = set(methods)
    return [method for method in METHOD_ORDER if method in available] + sorted(
        available - set(METHOD_ORDER)
    )


def _short(method: str) -> str:
    labels = {
        "local_only": "Local",
        "nearest_feasible_edge": "Nearest",
        "queue_energy_greedy": "Greedy",
        "genetic_drift": "Genetic",
        "hybrid_maddpg": "MADDPG",
        "domain_randomized_maddpg": "DR-MADDPG",
        "domain_randomized_no_queue_weight": "DR-MADDPG no-Q",
        "reptile_maddpg": "R-MADDPG",
        "reptile_no_queue_weight": "R-MADDPG no-Q",
        "hybrid_mappo": "MAPPO",
    }
    return labels.get(method, method.replace("_", " ").title())
