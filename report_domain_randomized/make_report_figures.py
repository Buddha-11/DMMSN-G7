"""Generate learned-method-only figures for the domain-randomized report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "output" / "studies" / "research-v1"
QUEUE_ABLATION_STUDY = ROOT / "output" / "studies" / "dr-queue-ablation-v1"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

METHODS = (
    "domain_randomized_maddpg",
    "hybrid_maddpg",
    "reptile_maddpg",
    "reptile_no_queue_weight",
    "hybrid_mappo",
)
LABELS = {
    "domain_randomized_maddpg": "QA-DR-MADDPG",
    "hybrid_maddpg": "Hybrid MADDPG",
    "reptile_maddpg": "Reptile-MADDPG",
    "reptile_no_queue_weight": "Reptile no queue",
    "hybrid_mappo": "Hybrid MAPPO",
}
COLORS = {
    "domain_randomized_maddpg": "#1565C0",
    "hybrid_maddpg": "#43A047",
    "reptile_maddpg": "#FB8C00",
    "reptile_no_queue_weight": "#8E24AA",
    "hybrid_mappo": "#546E7A",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean_by_method(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        method = row["method"]
        if method in METHODS:
            values[method].append(float(row[metric]))
    return {method: float(np.mean(values[method])) for method in METHODS}


def interval_by_method(
    rows: list[dict[str, str]], metric: str
) -> dict[str, tuple[float, float, float]]:
    intervals: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        if (
            row["method"] in METHODS
            and row["stress"] == "__all__"
            and row["metric"] == metric
        ):
            intervals[row["method"]] = (
                float(row["mean"]),
                float(row["lower"]),
                float(row["upper"]),
            )
    return intervals


def overall_performance(
    rows: list[dict[str, str]], intervals: list[dict[str, str]]
) -> None:
    metrics = (
        ("deadline_success_rate", "Deadline success", "Rate"),
        ("completion_rate", "Task completion", "Rate"),
        ("cumulative_reward", "Cumulative reward", "Reward"),
        ("energy_system_total_j", "System energy", "Energy (kJ)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    x = np.arange(len(METHODS))
    for axis, (metric, title, ylabel) in zip(axes.flat, metrics):
        means = mean_by_method(rows, metric)
        values = np.array([means[item] for item in METHODS])
        ci = interval_by_method(intervals, metric)
        lower = np.array([ci[item][1] for item in METHODS])
        upper = np.array([ci[item][2] for item in METHODS])
        if metric == "energy_system_total_j":
            values /= 1000.0
            lower /= 1000.0
            upper /= 1000.0
        errors = np.vstack((values - lower, upper - values))
        axis.bar(
            x,
            values,
            yerr=errors,
            capsize=3.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            color=[COLORS[item] for item in METHODS],
            width=0.72,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [LABELS[item] for item in METHODS], rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if "rate" in metric:
            axis.set_ylim(0.0, 1.1)
    figure.savefig(FIGURES / "overall_learned_methods.png", dpi=240)
    plt.close(figure)


def stress_heatmap(rows: list[dict[str, str]]) -> None:
    stresses = (
        "low_load",
        "near_capacity",
        "overload",
        "zero_harvest",
        "harvest_outage",
        "backhaul_bottleneck",
        "traffic_surge",
    )
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] in METHODS:
            values[(row["method"], row["stress"])].append(float(row["completion_rate"]))
    matrix = np.array(
        [[np.mean(values[(method, stress)]) for stress in stresses] for method in METHODS]
    )
    figure, axis = plt.subplots(figsize=(10.6, 4.5), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(stresses)), [item.replace("_", " ") for item in stresses], rotation=24, ha="right")
    axis.set_yticks(range(len(METHODS)), [LABELS[item] for item in METHODS])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            axis.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
    axis.set_title("Task completion across held-out stress conditions", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Completion rate")
    figure.savefig(FIGURES / "stress_completion_heatmap.png", dpi=240)
    plt.close(figure)


def domain_ablation(
    rows: list[dict[str, str]], intervals: list[dict[str, str]]
) -> None:
    methods = ("hybrid_maddpg", "domain_randomized_maddpg")
    metrics = (
        ("deadline_success_rate", "Deadline success", 1.0),
        ("completion_rate", "Completion", 1.0),
        ("throughput_tasks_per_s", "Throughput", 1.0),
        ("energy_system_total_j", "Energy (kJ)", 1000.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), constrained_layout=True)
    for axis, (metric, label, scale) in zip(axes.flat, metrics):
        means = mean_by_method(rows, metric)
        values = np.array([means[method] / scale for method in methods])
        ci = interval_by_method(intervals, metric)
        lower = np.array([ci[method][1] / scale for method in methods])
        upper = np.array([ci[method][2] / scale for method in methods])
        errors = np.vstack((values - lower, upper - values))
        axis.bar(
            [LABELS[method] for method in methods],
            values,
            yerr=errors,
            capsize=4,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            color=[COLORS[method] for method in methods],
            width=0.62,
        )
        axis.set_title(label, fontweight="bold")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        if metric in {"deadline_success_rate", "completion_rate"}:
            axis.set_ylim(0.0, 1.0)
    figure.suptitle("Domain-randomization ablation", fontweight="bold")
    figure.savefig(FIGURES / "domain_randomization_ablation.png", dpi=240)
    plt.close(figure)


def queue_weight_ablation(intervals: list[dict[str, str]]) -> None:
    methods = (
        "domain_randomized_no_queue_weight",
        "domain_randomized_maddpg",
    )
    labels = ("DR-MADDPG\nwithout queue weight", "QA-DR-MADDPG")
    colors = ("#9E9E9E", "#1565C0")
    metrics = (
        ("completion_rate", "Task completion", 100.0, "Completion (%)"),
        ("throughput_tasks_per_s", "Task throughput", 1.0, "Tasks/s"),
        ("final_backlog_cycles", "Final backlog", 1.0e-9, "Gcycles"),
        ("cumulative_reward", "Cumulative reward", 1.0e-3, "Reward ($\\times 10^3$)"),
    )
    lookup = {
        (row["method"], row["metric"]): (
            float(row["mean"]),
            float(row["lower"]),
            float(row["upper"]),
        )
        for row in intervals
        if row["stress"] == "__all__"
    }
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), constrained_layout=True)
    for axis, (metric, title, scale, ylabel) in zip(axes.flat, metrics):
        values = np.array([lookup[(method, metric)][0] * scale for method in methods])
        lower = np.array([lookup[(method, metric)][1] * scale for method in methods])
        upper = np.array([lookup[(method, metric)][2] * scale for method in methods])
        errors = np.vstack((values - lower, upper - values))
        axis.bar(
            labels,
            values,
            yerr=errors,
            capsize=4,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            color=colors,
            width=0.62,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "completion_rate":
            axis.set_ylim(0.0, 105.0)
    figure.suptitle("Direct physical-queue-weight ablation", fontweight="bold")
    figure.savefig(FIGURES / "queue_weight_ablation.png", dpi=240)
    plt.close(figure)


def network_and_slot_schematic() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    network, timeline = axes

    network.set_xlim(0, 10)
    network.set_ylim(0, 7)
    network.axis("off")
    network.set_title("Vehicle--RSU--UAV--cloud architecture", fontweight="bold")
    network.plot([0.3, 9.7], [0.8, 0.8], color="#707070", linewidth=2)

    for x, label in ((1.1, "Vehicle 1"), (3.1, "Vehicle 2"), (5.1, "Vehicle 3")):
        box = FancyBboxPatch(
            (x - 0.55, 0.95), 1.1, 0.55,
            boxstyle="round,pad=0.05", facecolor="#E3F2FD", edgecolor="#1565C0"
        )
        network.add_patch(box)
        network.text(x, 1.225, label, ha="center", va="center", fontsize=8)

    for x, label in ((2.0, "RSU 1"), (6.0, "RSU 2")):
        network.plot([x, x], [1.55, 3.15], color="#455A64", linewidth=2)
        network.plot([x - 0.25, x + 0.25], [3.15, 3.15], color="#455A64", linewidth=2)
        network.text(x, 3.42, label, ha="center", fontsize=8, fontweight="bold")

    for x, label in ((3.7, "UAV 1"), (7.0, "UAV 2")):
        network.add_patch(Circle((x, 5.1), 0.42, facecolor="#FFF3E0", edgecolor="#FB8C00", linewidth=1.5))
        network.plot([x - 0.65, x + 0.65], [5.1, 5.1], color="#FB8C00", linewidth=1.2)
        network.plot([x, x], [4.68, 5.52], color="#FB8C00", linewidth=1.2)
        network.text(x, 5.1, label, ha="center", va="center", fontsize=8, fontweight="bold")

    cloud = FancyBboxPatch(
        (8.1, 4.45), 1.45, 1.0, boxstyle="round,pad=0.08",
        facecolor="#F3E5F5", edgecolor="#8E24AA", linewidth=1.5
    )
    network.add_patch(cloud)
    network.text(8.825, 4.95, "Cloud\nserver", ha="center", va="center", fontsize=8, fontweight="bold")

    arrows = [
        ((1.1, 1.5), (2.0, 2.85)), ((3.1, 1.5), (3.7, 4.58)),
        ((5.1, 1.5), (6.0, 2.85)), ((5.1, 1.5), (7.0, 4.58)),
        ((3.7, 5.1), (8.05, 4.95)), ((7.0, 5.1), (8.05, 4.95)),
    ]
    for start, end in arrows:
        network.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10, color="#546E7A", linewidth=1.1))
    network.text(4.95, 6.25, "Mobile edge service and safe 3-D flight", ha="center", fontsize=8, color="#424242")

    timeline.set_xlim(0, 10)
    timeline.set_ylim(0, 5)
    timeline.axis("off")
    timeline.set_title("Causal order within slot $t$", fontweight="bold")
    stages = (
        ("Observe", "queues, tasks,\npositions, batteries"),
        ("Propose", "role-local\nactor outputs"),
        ("Repair", "route, CPU,\nflight, reserve"),
        ("Execute", "radio, compute,\nand motion"),
        ("Commit", "stage transfers\nat boundary"),
        ("Realize", "new tasks, harvest,\nmetrics"),
    )
    xs = np.linspace(0.9, 9.1, len(stages))
    colors = ("#E3F2FD", "#E8F5E9", "#FFF3E0", "#FCE4EC", "#EDE7F6", "#E0F7FA")
    for index, ((title, detail), x) in enumerate(zip(stages, xs)):
        box = FancyBboxPatch(
            (x - 0.66, 1.55), 1.32, 1.65,
            boxstyle="round,pad=0.06", facecolor=colors[index], edgecolor="#546E7A"
        )
        timeline.add_patch(box)
        timeline.text(x, 2.72, title, ha="center", va="center", fontsize=8, fontweight="bold")
        timeline.text(x, 2.18, detail, ha="center", va="center", fontsize=7)
        if index < len(stages) - 1:
            timeline.add_patch(FancyArrowPatch((x + 0.68, 2.38), (xs[index + 1] - 0.68, 2.38), arrowstyle="->", mutation_scale=10, color="#455A64"))
    timeline.text(5.0, 0.75, "Work completed in slot $t$ becomes eligible downstream in slot $t+1$", ha="center", fontsize=8, color="#424242")

    figure.savefig(FIGURES / "network_and_slot_schematic.png", dpi=260)
    plt.close(figure)


def learning_curves() -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), constrained_layout=True, sharey=True)
    for axis, method in zip(axes.flat, METHODS):
        method_root = STUDY / "training" / method
        seed_dirs = sorted(method_root.glob("seed-*"))
        seed_colors = plt.cm.viridis(np.linspace(0.10, 0.90, max(1, len(seed_dirs))))
        for seed_index, seed_dir in enumerate(seed_dirs):
            history = seed_dir / "history.jsonl"
            if not history.exists():
                continue
            records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
            x = [record["epoch"] for record in records if "validation_mean_reward_per_slot" in record]
            y = [record["validation_mean_reward_per_slot"] for record in records if "validation_mean_reward_per_slot" in record]
            if not y:
                continue
            values = np.asarray(y, dtype=float)
            # Edge padding keeps the first/last level from being pulled toward
            # zero while a short moving window suppresses single-epoch spikes.
            window = 7 if len(values) >= 15 else 5 if len(values) >= 7 else 3
            radius = window // 2
            padded = np.pad(values, (radius, radius), mode="edge")
            smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
            axis.plot(
                x,
                smoothed,
                linewidth=2.0,
                alpha=0.92,
                color=seed_colors[seed_index],
                solid_capstyle="round",
                solid_joinstyle="round",
            )
        axis.set_title(LABELS[method], fontsize=10, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Validation reward per slot")
    axes[1, 0].set_ylabel("Validation reward per slot")
    axes[1, 2].axis("off")
    figure.savefig(FIGURES / "learning_curves_learned_methods.png", dpi=240)
    plt.close(figure)


def main() -> None:
    rows = read_rows(STUDY / "learner_seed_metrics.csv")
    intervals = read_rows(STUDY / "learner_seed_confidence_intervals.csv")
    overall_performance(rows, intervals)
    stress_heatmap(rows)
    domain_ablation(rows, intervals)
    queue_weight_ablation(
        read_rows(QUEUE_ABLATION_STUDY / "learner_seed_confidence_intervals.csv")
    )
    learning_curves()
    network_and_slot_schematic()


if __name__ == "__main__":
    main()
