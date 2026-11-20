"""Executable gates for the independent hybrid-MAPPO VEC baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "hybrid MAPPO tests require PyTorch")
class HybridMAPPOTests(unittest.TestCase):
    @staticmethod
    def manifest():
        from src.vec.evaluation import create_standard_manifest

        return create_standard_manifest(
            {"train": (0,), "validation": (100,), "test": (200,)},
            horizon_slots=4,
            drain_slots=2,
            manifest_id="mappo-unit-v1",
        )

    @staticmethod
    def config():
        from src.vec.training import MAPPOTrainingConfig

        return MAPPOTrainingConfig(
            epochs=1,
            scenarios_per_epoch=1,
            validation_scenarios=1,
            actor_hidden_dim=16,
            critic_hidden_dim=24,
            ppo_epochs=2,
            mini_batches=2,
            early_stopping_patience=2,
        )

    def test_policy_samples_only_masked_and_bounded_hybrid_actions(self):
        from src.vec.evaluation import instantiate_scenario
        from src.vec.learning import HybridMAPPO, HybridMAPPOConfig, VecPolicyAdapter

        scenario = instantiate_scenario(self.manifest().select(split="train")[0])
        environment = scenario.environment()
        adapter = VecPolicyAdapter(environment, max_tasks=4)
        encoded = adapter.encode()
        learner = HybridMAPPO(
            HybridMAPPOConfig(
                state_dim=len(encoded.state),
                role_specs=adapter.role_specs(),
                actor_hidden_dim=16,
                critic_hidden_dim=24,
            )
        )
        actions, latents, log_probabilities = learner.act(
            encoded.observations,
            encoded.task_features,
            encoded.task_masks,
            encoded.categorical_masks,
        )
        vehicle = actions["vehicle"]
        category_count = learner.role_specs[0].categorical_dim
        self.assertTrue(np.allclose(vehicle[:, :category_count].sum(axis=1), 1.0))
        chosen = vehicle[:, :category_count].argmax(axis=1)
        for agent, category in enumerate(chosen):
            self.assertTrue(encoded.categorical_masks["vehicle"][agent, category])
        self.assertTrue(np.all(np.abs(actions["uav"][:, 1:4]) <= 1.0))
        for role in latents:
            self.assertTrue(np.isfinite(latents[role]).all())
            self.assertTrue(np.isfinite(log_probabilities[role]).all())

    def test_training_checkpoint_resume_and_common_evaluator(self):
        from src.vec.evaluation import EvaluationMethod, evaluate_scenario, instantiate_scenario
        from src.vec.learning import HybridMAPPOController
        from src.vec.training import VecMAPPOTrainer

        manifest = self.manifest()
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            result = VecMAPPOTrainer(config, manifest, directory).train()
            root = Path(directory)
            self.assertEqual(result.completed_epochs, 1)
            self.assertTrue((root / "best.pt").is_file())
            self.assertTrue((root / "artifact_manifest.json").is_file())
            checkpoint = root / "best.pt"
            scenario = instantiate_scenario(manifest.select(split="test")[0])
            method = EvaluationMethod(
                "hybrid_mappo",
                "learned_on_policy",
                lambda instance, environment: HybridMAPPOController.from_checkpoint(
                    "hybrid_mappo", checkpoint, environment
                ),
                checkpoint=str(checkpoint),
            )
            run = evaluate_scenario(scenario, method)
            self.assertEqual(run.method, "hybrid_mappo")
            self.assertGreater(len(run.traces), 0)
            resumed = VecMAPPOTrainer(config, manifest, directory).train(resume=True)
            self.assertEqual(resumed.completed_epochs, 1)
            self.assertEqual(len((root / "history.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
