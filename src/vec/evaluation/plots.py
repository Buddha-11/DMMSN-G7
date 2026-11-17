"""Presentation-ready figures generated only from saved evaluation runs."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from statistics import fmean
from typing import Iterable

import numpy as np

from .runner import EvaluationRun
from .statistics import confidence_interval


def generate_figures(
    runs: Iterable[EvaluationRun], output_directory: str | Path
) -> tuple[Path, ...]:
    output = Path(output_directory)
    matplotlib_config = output.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib.pyplot as plt

    runs = tuple(runs)
    if not runs:
        raise ValueError("cannot plot an empty evaluation")
    output.mkdir(parents=True, exist_ok=True)
    paths = (
        output / "headline_metrics.png",
        output / "backlog_trajectories.png",
        output / "adaptation_curves.png",
    )

    _headline_plot(plt, runs, paths[0])
    _backlog_plot(plt, runs, paths[1])
    _adaptation_plot(plt, runs, paths[2])
    return paths


def _headline_plot(plt, runs: tuple[EvaluationRun, ...], path: Path) -> None:
    methods = sorted({run.method for run in runs})
    fields = (
        ("deadline_success_rate", "Deadline success", True),
        ("throughput_tasks_per_s", "Throughput (tasks/s)", True),
        ("final_backlog_cycles", "Final backlog (cycles)", False),
        ("controller_mean_seconds", "Decision time (s)", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (field, title, _) in zip(axes.flat, fields, strict=True):
        means = []
        errors = []
        for method in methods:
            # Macro-average stress suites inside each independent seed first;
            # otherwise treating every stress/seed row as independent would
            # understate uncertainty through pseudoreplication.
            by_seed: dict[int, list[float]] = defaultdict(list)
            for run in runs:
                if run.method == method:
                    by_seed[run.seed].append(float(getattr(run.metrics, field)))
            values = [fmean(seed_values) for seed_values in by_seed.values()]
            interval = confidence_interval(values)
            means.append(interval.mean)
            errors.append(interval.upper - interval.mean)
        axis.bar(methods, means, yerr=errors, capsize=3)
        axis.set_title(f"{title} (stress-macro mean, 95% seed CI)")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _backlog_plot(plt, runs: tuple[EvaluationRun, ...], path: Path) -> None:
    stresses = sorted({run.stress for run in runs})
    methods = sorted({run.method for run in runs})
    columns = 2
    rows = (len(stresses) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(13, 3.6 * rows), squeeze=False, constrained_layout=True
    )
    for axis, stress in zip(axes.flat, stresses):
        for method in methods:
            selected = [
                run for run in runs if run.stress == stress and run.method == method
            ]
            if not selected:
                continue
            length = min(len(run.traces) for run in selected)
            values = np.asarray(
                [
                    [trace.total_backlog_cycles for trace in run.traces[:length]]
                    for run in selected
                ],
                dtype=float,
            )
            mean = values.mean(axis=0)
            if len(selected) > 1:
                error = values.std(axis=0, ddof=1) / np.sqrt(len(selected))
                error *= 1.96
            else:
                error = np.zeros(length)
            slots = np.arange(length)
            axis.plot(slots, mean, label=method)
            axis.fill_between(slots, mean - error, mean + error, alpha=0.15)
        axis.set_title(stress.replace("_", " ").title())
        axis.set_xlabel("Slot")
        axis.set_ylabel("Backlog (cycles)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(stresses) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _adaptation_plot(plt, runs: tuple[EvaluationRun, ...], path: Path) -> None:
    selected = [run for run in runs if run.stress == "traffic_surge"]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in selected:
        for offset, reward in run.metrics.adaptation_curve_reward.items():
            grouped[run.method][int(offset)].append(reward)
    for method, by_offset in sorted(grouped.items()):
        offsets = sorted(by_offset)
        means = [fmean(by_offset[offset]) for offset in offsets]
        axis.plot(offsets, means, marker="o", label=method)
    axis.set_title("Post-change reward adaptation curve")
    axis.set_xlabel("Slots since declared change")
    axis.set_ylabel("Mean reward through offset")
    axis.grid(alpha=0.25)
    if grouped:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No traffic-surge runs", ha="center", va="center")
    figure.savefig(path, dpi=200)
    plt.close(figure)
