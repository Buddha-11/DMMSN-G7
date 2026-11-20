#!/usr/bin/env python3
"""Generate human-readable and IEEE-LaTeX summaries from a completed VEC study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD_ORDER = (
    "local_only",
    "nearest_feasible_edge",
    "queue_energy_greedy",
    "genetic_drift",
    "hybrid_maddpg",
    "domain_randomized_maddpg",
    "reptile_maddpg",
    "reptile_no_queue_weight",
    "hybrid_mappo",
)
LABELS = {
    "local_only": "Local only",
    "nearest_feasible_edge": "Nearest edge",
    "queue_energy_greedy": "Queue-energy greedy",
    "genetic_drift": "Genetic drift",
    "hybrid_maddpg": "Hybrid MADDPG",
    "domain_randomized_maddpg": "DR-MADDPG",
    "reptile_maddpg": "R-MADDPG",
    "reptile_no_queue_weight": "R-MADDPG no-Q",
    "hybrid_mappo": "Hybrid MAPPO",
}
STRESSES = (
    "low_load",
    "near_capacity",
    "overload",
    "zero_harvest",
    "harvest_outage",
    "backhaul_bottleneck",
    "traffic_surge",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="output/studies/research-v1")
    parser.add_argument("--markdown", default="RESULTS_SUMMARY.md")
    parser.add_argument(
        "--latex", default="report/IEEEtran/results_generated.tex"
    )
    args = parser.parse_args()

    root = Path(args.study).resolve()
    summary = json.loads((root / "study_summary.json").read_text(encoding="utf-8"))
    intervals = _read_csv(root / "learner_seed_confidence_intervals.csv")
    pairwise = _read_csv(root / "learner_seed_pairwise_comparisons.csv")
    runs = _read_csv(root / "study_run_metrics.csv")
    index = {
        (row["method"], row["stress"], row["metric"]): row for row in intervals
    }
    methods = [method for method in METHOD_ORDER if method in summary["methods"]]

    markdown_path = Path(args.markdown)
    markdown_path.write_text(
        _markdown(root, summary, methods, index, pairwise, runs),
        encoding="utf-8",
    )
    latex_path = Path(args.latex)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(_latex(methods, index), encoding="utf-8")
    print(markdown_path.resolve())
    print(latex_path.resolve())
    return 0


def _markdown(root, summary, methods, index, pairwise, runs) -> str:
    lines = [
        "# Research-scale VEC results",
        "",
        f"Study: `{root}`",
        "",
        f"Research-claim protocol ready: **{summary['research_claim_ready']}**  ",
        f"Independent learner seeds: **{summary['learner_seed_count']}**  ",
        f"Paired held-out test runs: **{summary['test_run_count']}**  ",
        f"Safety valid: **{summary['safety_valid']}**",
        "",
        "All confidence intervals below use the independent learner seed as the sampling unit after macro-averaging paired test scenarios. Latency covers completed tasks only and must be read with completion/censoring.",
        "",
        "## Overall held-out performance",
        "",
        "| Method | Completion (%) | Deadline success (%) | Final backlog | System energy | p95 latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            "| "
            + " | ".join(
                (
                    LABELS[method],
                    _ci(index, method, "__all__", "completion_rate", percent=True),
                    _ci(index, method, "__all__", "deadline_success_rate", percent=True),
                    _ci(index, method, "__all__", "final_backlog_cycles", scale=1e9, unit=" Gcycle"),
                    _ci(index, method, "__all__", "energy_system_total_j", scale=1e3, unit=" kJ"),
                    _ci(index, method, "__all__", "response_time_p95_s", unit=" s"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Completion rate by stress",
            "",
            "| Stress | Greedy | R-MADDPG | R-MADDPG no-Q | MAPPO |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stress in STRESSES:
        lines.append(
            "| "
            + " | ".join(
                (
                    stress.replace("_", " ").title(),
                    _mean(index, "queue_energy_greedy", stress, "completion_rate", percent=True),
                    _mean(index, "reptile_maddpg", stress, "completion_rate", percent=True),
                    _mean(index, "reptile_no_queue_weight", stress, "completion_rate", percent=True),
                    _mean(index, "hybrid_mappo", stress, "completion_rate", percent=True),
                )
            )
            + " |"
        )
    lines.extend(["", "## Direct paired comparisons", ""])
    for other in (
        "queue_energy_greedy",
        "hybrid_maddpg",
        "domain_randomized_maddpg",
        "reptile_no_queue_weight",
        "hybrid_mappo",
    ):
        comparison = _normalized_comparison(
            pairwise, "reptile_maddpg", other, "__all__", "completion_rate"
        )
        if comparison is None:
            continue
        lines.append(
            f"- R-MADDPG minus {LABELS[other]} completion rate: "
            f"{comparison[0] * 100:.2f} percentage points "
            f"(95% paired CI {comparison[1] * 100:.2f} to {comparison[2] * 100:.2f}; "
            f"n={comparison[3]} learner seeds)."
        )
    safety = {
        metric: sum(float(row[metric]) for row in runs)
        for metric in (
            "physical_feasibility_failures",
            "battery_violations",
            "separation_violations",
        )
    }
    lines.extend(
        [
            "",
            "## Safety and limitations",
            "",
            f"- Physical feasibility failures: {safety['physical_feasibility_failures']:.0f}.",
            f"- Battery violations: {safety['battery_violations']:.0f}.",
            f"- Separation violations: {safety['separation_violations']:.0f}.",
            "- The frozen research study uses procedurally generated road mobility, task arrivals, and energy harvesting. It does not constitute a Bologna/SUMO real-road result.",
            "- Overload backlog growth is an expected infeasibility indicator, not evidence of universal stability.",
            "- A confidence interval crossing zero does not support a directional superiority claim.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe scripts\run_vec_study.py --output output\studies\research-v1 --workers 4",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _latex(methods, index) -> str:
    lines = [
        "% Generated by scripts/generate_vec_results.py; do not hand-edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Held-out test performance, reported as learner-seed mean with 95\\% confidence interval.}",
        "\\label{tab:main_results}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Method & Completion (\\%) & Deadline (\\%) & Final backlog (Gcycle) & Energy (kJ)\\\\",
        "\\hline",
    ]
    for method in methods:
        label = LABELS[method]
        values = (
            _latex_ci(index, method, "completion_rate", scale=100.0),
            _latex_ci(index, method, "deadline_success_rate", scale=100.0),
            _latex_ci(index, method, "final_backlog_cycles", scale=1.0 / 1e9),
            _latex_ci(index, method, "energy_system_total_j", scale=1.0 / 1e3),
        )
        lines.append(f"{label} & " + " & ".join(values) + r"\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _normalized_comparison(rows, proposed, other, stress, metric):
    for row in rows:
        if row["stress"] != stress or row["metric"] != metric:
            continue
        if row["method"] == proposed and row["reference_method"] == other:
            return (
                float(row["mean_paired_difference"]),
                float(row["difference_ci_lower"]),
                float(row["difference_ci_upper"]),
                int(row["paired_count"]),
            )
        if row["method"] == other and row["reference_method"] == proposed:
            return (
                -float(row["mean_paired_difference"]),
                -float(row["difference_ci_upper"]),
                -float(row["difference_ci_lower"]),
                int(row["paired_count"]),
            )
    return None


def _ci(index, method, stress, metric, *, percent=False, scale=1.0, unit=""):
    row = index.get((method, stress, metric))
    if row is None:
        return "n/a"
    factor = 100.0 if percent else 1.0 / scale
    return (
        f"{float(row['mean']) * factor:.2f} "
        f"[{float(row['lower']) * factor:.2f}, {float(row['upper']) * factor:.2f}]{unit}"
    )


def _mean(index, method, stress, metric, *, percent=False):
    row = index.get((method, stress, metric))
    if row is None:
        return "n/a"
    factor = 100.0 if percent else 1.0
    return f"{float(row['mean']) * factor:.2f}" + ("%" if percent else "")


def _latex_ci(index, method, metric, *, scale):
    row = index.get((method, "__all__", metric))
    if row is None:
        return "--"
    mean = float(row["mean"]) * scale
    half = (float(row["upper"]) - float(row["mean"])) * scale
    return f"{mean:.2f} $\\pm$ {half:.2f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    raise SystemExit(main())
