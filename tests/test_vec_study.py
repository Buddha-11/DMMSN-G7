"""Tests for the complete multi-seed experiment layer."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.vec.evaluation.runner import REQUIRED_COMPARISON_METHODS
from src.vec.evaluation.study_plots import generate_study_figures
from src.vec.experiments import (
    DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS,
    LEARNED_METHODS,
    OPTIONAL_STUDY_METRICS,
    SAFETY_METRICS,
    StudyConfig,
    STUDY_METRICS,
    VecStudyRunner,
    aggregate_study_evaluations,
)


class StudyPlanTests(unittest.TestCase):
    def test_plan_uses_one_manifest_and_every_method_seed_pair(self):
        config = StudyConfig(
            learner_seeds=(3, 7),
            train_seeds=(10,),
            validation_seeds=(110,),
            test_seeds=(210,),
            epochs=2,
            scenarios_per_epoch=1,
            horizon_slots=5,
            drain_slots=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = VecStudyRunner(
                config,
                directory,
                project_root=Path(__file__).resolve().parents[1],
                python_executable=sys.executable,
            )
            jobs = runner.prepare()
            self.assertEqual(len(jobs), len(LEARNED_METHODS) * 2)
            self.assertEqual(
                {(job.method, job.learner_seed) for job in jobs},
                {(method, seed) for method in LEARNED_METHODS for seed in (3, 7)},
            )
            manifest_paths = {
                job.command[job.command.index("--manifest") + 1] for job in jobs
            }
            self.assertEqual(manifest_paths, {str(runner.manifest_path)})
            plan = json.loads(
                (Path(directory) / "study_plan.json").read_text(encoding="utf-8")
            )
            self.assertIn("independent learner seed", plan["statistical_unit"])

    def test_domain_randomized_queue_ablation_schedules_only_two_variants(self):
        config = StudyConfig(
            learner_seeds=(3, 7),
            train_seeds=(10,),
            validation_seeds=(110,),
            test_seeds=(210,),
            epochs=2,
            scenarios_per_epoch=1,
            horizon_slots=5,
            drain_slots=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = VecStudyRunner(
                config,
                directory,
                project_root=Path(__file__).resolve().parents[1],
                python_executable=sys.executable,
                learned_methods=DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS,
                baseline_methods=(),
                require_complete_roster=False,
            )
            jobs = runner.prepare()
            self.assertEqual(len(jobs), 4)
            self.assertEqual(
                {job.method for job in jobs},
                set(DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS),
            )
            for job in jobs:
                self.assertIn("domain_randomized", job.command)
                self.assertNotIn("reptile", job.command)
                self.assertNotIn("hybrid_mappo", job.command)
                if job.method == "domain_randomized_no_queue_weight":
                    self.assertIn("--no-queue-weight", job.command)
                else:
                    self.assertNotIn("--no-queue-weight", job.command)
            for method in DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS:
                training = (
                    Path(directory) / "training" / method / "seed-003"
                )
                training.mkdir(parents=True)
                (training / "training_summary.json").write_text(
                    json.dumps({"offline_training_seconds": 1.0}),
                    encoding="utf-8",
                )
            evaluation = runner._evaluation_command(3, plots=False)
            self.assertNotIn("--require-complete-roster", evaluation)
            self.assertEqual(evaluation[evaluation.index("--methods") + 1], "")
            checkpoints = [
                evaluation[index + 1].split("=", 1)[0]
                for index, value in enumerate(evaluation)
                if value == "--checkpoint"
            ]
            self.assertEqual(
                set(checkpoints), set(DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS)
            )

    def test_scenario_splits_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            StudyConfig(
                learner_seeds=(0,),
                train_seeds=(1,),
                validation_seeds=(2,),
                test_seeds=(1,),
            )

    def test_study_manifest_records_checkpoints_runtime_seeds_and_data_license(self):
        config = StudyConfig(
            learner_seeds=(3,),
            train_seeds=(10,),
            validation_seeds=(110,),
            test_seeds=(210,),
            epochs=1,
            scenarios_per_epoch=1,
            horizon_slots=5,
            drain_slots=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = VecStudyRunner(
                config,
                root,
                project_root=Path(__file__).resolve().parents[1],
                python_executable=sys.executable,
            )
            runner.prepare()
            checkpoint = root / "training" / "hybrid_maddpg" / "seed-003" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"selected checkpoint")
            evaluation_manifest = (
                root / "evaluation" / "seed-003" / "artifact_manifest.json"
            )
            evaluation_manifest.parent.mkdir(parents=True)
            evaluation_manifest.write_text("{}\n", encoding="utf-8")

            runner._write_artifact_manifest()
            payload = json.loads(
                (root / "study_artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["schema_version"], "vec-study-artifacts/v2")
            self.assertEqual(payload["seeds"]["learner"], [3])
            self.assertIn(
                "training/hybrid_maddpg/seed-003/best.pt",
                payload["selected_checkpoints"]["files"],
            )
            self.assertIn("torch", payload["runtime"]["dependencies"])
            self.assertFalse(payload["datasets"]["external_datasets_used"])
            self.assertIn(
                "no external dataset",
                payload["datasets"]["generated_sources"][0]["license"],
            )


class StudyAggregationTests(unittest.TestCase):
    def test_focused_ablation_aggregates_without_full_benchmark_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation" / "seed-000"
            evaluation.mkdir(parents=True)
            methods = DOMAIN_RANDOMIZED_QUEUE_ABLATION_METHODS
            (evaluation / "evaluation_status.json").write_text(
                json.dumps(
                    {
                        "comparison_complete": False,
                        "trace_pairing_valid": True,
                        "methods_run": list(methods),
                    }
                ),
                encoding="utf-8",
            )
            rows = []
            for index, method in enumerate(methods):
                rows.append(
                    {
                        "scenario_id": "test-low_load-s200",
                        "stress": "low_load",
                        "method": method,
                        **{
                            metric: 0.0 if metric in SAFETY_METRICS else index + 0.5
                            for metric in STUDY_METRICS
                        },
                        **{
                            metric: index + 0.5
                            for metric in OPTIONAL_STUDY_METRICS
                        },
                    }
                )
            with (evaluation / "run_metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            aggregate_study_evaluations(
                root,
                (0,),
                reference_method=methods[0],
                smoke=True,
                required_methods=methods,
                require_complete_roster=False,
            )
            summary = json.loads(
                (root / "study_summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["comparison_complete"])
            self.assertTrue(summary["requested_comparison_complete"])
            self.assertFalse(summary["require_complete_roster"])
            with (root / "learner_seed_paired_comparisons.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                comparisons = list(csv.DictReader(stream))
            self.assertTrue(
                any(
                    row["method"] == methods[1]
                    and row["reference_method"] == methods[0]
                    for row in comparisons
                )
            )

    def test_aggregation_uses_learner_seed_as_statistical_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for learner_seed in (0, 1):
                evaluation = root / "evaluation" / f"seed-{learner_seed:03d}"
                evaluation.mkdir(parents=True)
                (evaluation / "evaluation_status.json").write_text(
                    json.dumps(
                        {"comparison_complete": True, "trace_pairing_valid": True}
                    ),
                    encoding="utf-8",
                )
                rows = []
                for method_index, method in enumerate(REQUIRED_COMPARISON_METHODS):
                    for scenario_seed in (200, 201):
                        base = learner_seed + method_index + scenario_seed / 1000.0
                        rows.append(
                            {
                                "scenario_id": f"test-low_load-s{scenario_seed}",
                                "stress": "low_load",
                                "method": method,
                                **{
                                    metric: 0.0 if metric in SAFETY_METRICS else base
                                    for metric in STUDY_METRICS
                                },
                                **{
                                    metric: base
                                    for metric in OPTIONAL_STUDY_METRICS
                                },
                            }
                        )
                with (evaluation / "run_metrics.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            aggregate_study_evaluations(root, (0, 1), smoke=True)
            with (root / "learner_seed_confidence_intervals.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                intervals = list(csv.DictReader(stream))
            completion = next(
                row
                for row in intervals
                if row["method"] == "hybrid_maddpg"
                and row["stress"] == "low_load"
                and row["metric"] == "completion_rate"
            )
            self.assertEqual(int(completion["count"]), 2)
            response_tail = next(
                row
                for row in intervals
                if row["method"] == "hybrid_maddpg"
                and row["stress"] == "low_load"
                and row["metric"] == "response_time_p95_s"
            )
            self.assertEqual(int(response_tail["count"]), 2)
            with (root / "learner_seed_pairwise_comparisons.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                pairwise = list(csv.DictReader(stream))
            self.assertTrue(
                any(
                    row["metric"] == "completion_rate"
                    and row["stress"] == "__all__"
                    for row in pairwise
                )
            )
            summary = json.loads(
                (root / "study_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["comparison_complete"])
            self.assertTrue(summary["safety_valid"])
            self.assertFalse(summary["protocol_scale_valid"])
            self.assertFalse(summary["research_claim_ready"])
            self.assertEqual(summary["learner_seed_count"], 2)
            figures = generate_study_figures(root)
            self.assertEqual(len(figures), 4)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in figures))


if __name__ == "__main__":
    unittest.main()
