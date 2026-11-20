"""Gate tests for reproducible, research-defensible Phase 6 evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.vec.evaluation import (
    EvidenceWriter,
    STANDARD_STRESSES,
    ScenarioManifest,
    ScenarioSpec,
    baseline_methods,
    confidence_interval,
    create_standard_manifest,
    evaluate_scenario,
    instantiate_scenario,
    load_recorded_mobility_csv,
    missing_required_methods,
    paired_comparison,
)


class ScenarioManifestTests(unittest.TestCase):
    def test_standard_suite_is_deterministic_and_split_by_complete_scenario(self):
        manifest = create_standard_manifest(
            {"train": (1,), "validation": (101,), "test": (201,)},
            horizon_slots=5,
            drain_slots=2,
        )
        self.assertEqual(
            {item.stress for item in manifest.select(split="test")},
            set(STANDARD_STRESSES),
        )
        first = instantiate_scenario(manifest.select(split="test")[0])
        second = instantiate_scenario(manifest.select(split="test")[0])
        self.assertEqual(first.trace_digest, second.trace_digest)
        self.assertEqual(first.arrivals, second.arrivals)
        self.assertEqual(first.harvests, second.harvests)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.save(path)
            self.assertEqual(ScenarioManifest.load(path), manifest)

    def test_seed_leakage_across_splits_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cross manifest splits"):
            ScenarioManifest(
                "bad",
                (
                    ScenarioSpec("train-a", "train", "low_load", 7),
                    ScenarioSpec("test-a", "test", "low_load", 7),
                ),
            )

    def test_recorded_sumo_export_requires_complete_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sumo_positions.csv"
            path.write_text(
                "slot,vehicle_id,x,y,z\n"
                "0,v0,0,0,0\n0,v1,1,0,0\n0,v2,2,0,0\n"
                "1,v0,1,0,0\n1,v1,2,0,0\n1,v2,3,0,0\n",
                encoding="utf-8",
            )
            frames = load_recorded_mobility_csv(path, ("v0", "v1", "v2"))
            self.assertEqual(frames[1][2], (3.0, 0.0, 0.0))
            scenario = instantiate_scenario(
                ScenarioSpec(
                    "test-recorded",
                    "test",
                    "low_load",
                    501,
                    horizon_slots=2,
                    drain_slots=1,
                    mobility_trace=str(path),
                    mobility_vehicle_ids=("v0", "v1", "v2"),
                )
            )
            environment = scenario.environment()
            self.assertEqual(environment.vehicle_positions, frames[0])
            environment.step()
            self.assertEqual(environment.vehicle_positions, frames[1])

    def test_zero_harvest_suite_returns_and_lands_without_physical_failure(self):
        scenario = instantiate_scenario(
            ScenarioSpec(
                "test-zero-harvest-return",
                "test",
                "zero_harvest",
                777,
                horizon_slots=200,
                drain_slots=100,
                mean_harvest_j=0.0,
            )
        )
        method = next(
            item for item in baseline_methods() if item.name == "local_only"
        )
        run = evaluate_scenario(scenario, method)

        self.assertEqual(run.metrics.physical_feasibility_failures, 0)
        self.assertEqual(run.metrics.battery_violations, 0)
        self.assertGreater(run.metrics.landing_uav_slots, 0)


class StatisticalEvidenceTests(unittest.TestCase):
    def test_confidence_interval_and_paired_effect_use_seed_level_samples(self):
        interval = confidence_interval((1.0, 2.0, 3.0, 4.0))
        self.assertEqual(interval.count, 4)
        self.assertAlmostEqual(interval.mean, 2.5)
        self.assertLess(interval.lower, interval.mean)
        self.assertGreater(interval.upper, interval.mean)

        comparison = paired_comparison(
            {1: 4.0, 2: 5.0, 3: 7.0},
            {1: 3.0, 2: 5.0, 3: 6.0},
            method="new",
            reference_method="old",
            metric="throughput",
            higher_is_better=True,
        )
        self.assertEqual(comparison.paired_count, 3)
        self.assertEqual((comparison.wins, comparison.ties, comparison.losses), (2, 1, 0))

    def test_unpaired_scenario_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical scenario keys"):
            paired_comparison(
                {1: 1.0},
                {2: 1.0},
                method="a",
                reference_method="b",
                metric="x",
                higher_is_better=True,
            )


class EvaluationPipelineTests(unittest.TestCase):
    @staticmethod
    def scenario(seed=301):
        return instantiate_scenario(
            ScenarioSpec(
                scenario_id=f"test-overload-s{seed}",
                split="test",
                stress="overload",
                seed=seed,
                horizon_slots=4,
                drain_slots=2,
                nominal_load_ratio=1.25,
            )
        )

    def test_run_reports_censoring_stage_slopes_energy_and_violations(self):
        scenario = self.scenario()
        method = next(
            item for item in baseline_methods() if item.name == "local_only"
        )
        run = evaluate_scenario(scenario, method)

        self.assertEqual(run.trace_digest, scenario.trace_digest)
        self.assertEqual(run.metrics.scheduled_tasks, len(scenario.arrivals))
        self.assertEqual(
            run.metrics.completed_tasks + run.metrics.unfinished_censored_tasks,
            run.metrics.admitted_tasks,
        )
        self.assertIn("ingress", run.metrics.backlog_by_stage_final_cycles)
        self.assertIn("system_total", run.metrics.energy_j)
        self.assertIn("uav_propulsion", run.metrics.energy_j)
        self.assertIn("uav_computation", run.metrics.energy_j)
        self.assertIn("uav_reception", run.metrics.energy_j)
        self.assertIn("uav_backhaul", run.metrics.energy_j)
        self.assertIn("uav_control", run.metrics.energy_j)
        self.assertGreaterEqual(run.metrics.response_time_p95_s or 0.0, 0.0)
        self.assertEqual(run.metrics.battery_violations, 0)
        self.assertEqual(run.metrics.conservation_violations, 0)
        self.assertGreater(run.metrics.controller_total_seconds, 0.0)

    def test_paired_methods_share_exact_exogenous_trace_and_artifacts_are_hashed(self):
        scenario = self.scenario(seed=302)
        selected = tuple(
            method
            for method in baseline_methods()
            if method.name in {"local_only", "queue_energy_greedy"}
        )
        runs = tuple(evaluate_scenario(scenario, method) for method in selected)
        self.assertEqual(len({run.trace_digest for run in runs}), 1)

        manifest = ScenarioManifest("test", (scenario.spec,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = EvidenceWriter(
                root,
                manifest,
                selected,
                command="python scripts/evaluate_vec.py --example",
            )
            writer.write_scenario(scenario)
            for run in runs:
                writer.write_run(run)
            writer.write_aggregate(runs)

            status = json.loads(
                (root / "evaluation_status.json").read_text(encoding="utf-8")
            )
            hashes = json.loads(
                (root / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(status["trace_pairing_valid"])
            self.assertFalse(status["comparison_complete"])
            self.assertIn("run_metrics.csv", hashes["files"])
            with (root / "run_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                metrics_rows = list(csv.DictReader(stream))
            self.assertIn("energy_uav_propulsion_j", metrics_rows[0])
            self.assertIn("minimum_battery_charge_j", metrics_rows[0])
            self.assertIn("controller_p95_seconds", metrics_rows[0])
            with (root / "paired_comparisons.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                comparisons = list(csv.DictReader(stream))
            self.assertTrue(comparisons)

    def test_required_algorithm_roster_cannot_be_silently_claimed(self):
        missing = missing_required_methods(baseline_methods())
        self.assertIn("hybrid_maddpg", missing)
        self.assertIn("reptile_maddpg", missing)
        self.assertIn("hybrid_mappo", missing)


if __name__ == "__main__":
    unittest.main()
