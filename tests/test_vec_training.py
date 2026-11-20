"""Executable gates for the reproducible policy-training orchestrator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "training tests require PyTorch")
class TrainingPipelineTests(unittest.TestCase):
    def test_exploration_preserves_masks_bounds_and_share_simplex(self):
        from src.vec.learning import RoleSpec
        from src.vec.training import explore_hybrid_actions

        specs = (
            RoleSpec(
                "vehicle", 1, 2, 2, 3, categorical_dim=4, has_split=True
            ),
            RoleSpec(
                "uav", 1, 2, 2, 3, has_cpu=True, has_motion=True
            ),
        )
        actions = {
            "vehicle": np.asarray([[1, 0, 0, 0, 0.5]], dtype=np.float32),
            "uav": np.asarray([[0.5, 0, 0, 0, 0.3, 0.7, 0]], dtype=np.float32),
        }
        masks = {
            "vehicle": np.asarray([[True, False, False]]),
            "uav": np.asarray([[True, True, False]]),
        }
        categorical = {
            "vehicle": np.asarray([[False, True, False, True]]),
            "uav": np.zeros((1, 0), dtype=np.bool_),
        }
        explored = explore_hybrid_actions(
            specs,
            actions,
            masks,
            categorical,
            standard_deviation=1.0,
            random=np.random.default_rng(4),
        )
        self.assertEqual(float(explored["vehicle"][0, :4].sum()), 1.0)
        self.assertFalse(bool(explored["vehicle"][0, 0]))
        self.assertTrue(np.all(np.abs(explored["uav"][0, 1:4]) <= 1.0))
        self.assertAlmostEqual(float(explored["uav"][0, 4:].sum()), 1.0)
        self.assertEqual(float(explored["uav"][0, -1]), 0.0)

    def test_smoke_training_writes_auditable_checkpoint_bundle(self):
        from src.vec.evaluation import create_standard_manifest
        from src.vec.training import TrainingConfig, VecTrainer

        manifest = create_standard_manifest(
            {"train": (0,), "validation": (100,), "test": (200,)},
            horizon_slots=3,
            drain_slots=1,
            manifest_id="unit-training-v1",
        )
        config = TrainingConfig(
            mode="domain_randomized",
            epochs=1,
            scenarios_per_epoch=1,
            validation_scenarios=1,
            actor_hidden_dim=8,
            critic_hidden_dim=12,
            batch_size=2,
            replay_capacity=16,
            warmup_steps=2,
            max_tasks=4,
            early_stopping_patience=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = VecTrainer(config, manifest, directory).train()
            root = Path(directory)
            self.assertEqual(result.completed_epochs, 1)
            for name in (
                "best.pt",
                "last.pt",
                "training_config.json",
                "scenario_manifest.json",
                "provenance.json",
                "history.jsonl",
                "training_state.pt",
                "training_summary.json",
                "artifact_manifest.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            self.assertNotIn("test-", (root / "history.jsonl").read_text())
            resumed = VecTrainer(config, manifest, directory).train(resume=True)
            self.assertEqual(resumed.completed_epochs, 1)
            self.assertEqual(
                len((root / "history.jsonl").read_text().splitlines()), 1
            )


if __name__ == "__main__":
    unittest.main()
