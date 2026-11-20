"""Executable gate tests for Phase 5 hybrid MADDPG and Reptile."""

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


@unittest.skipUnless(TORCH_AVAILABLE, "Phase 5 tests require PyTorch")
class Phase5LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.vec.learning import RoleSpec

        cls.RoleSpec = RoleSpec

    def config(self, *, state_dim=5, batch_size=2, seed=7):
        from src.vec.learning import HybridMADDPGConfig

        return HybridMADDPGConfig(
            state_dim=state_dim,
            role_specs=(
                self.RoleSpec("vehicle", 2, 3, 4, 3, 4, has_split=True),
                self.RoleSpec("rsu", 1, 2, 4, 3, has_cpu=True),
                self.RoleSpec(
                    "uav", 1, 4, 4, 3, has_cpu=True, has_motion=True
                ),
            ),
            actor_hidden_dim=16,
            critic_hidden_dim=24,
            batch_size=batch_size,
            replay_capacity=16,
            seed=seed,
        )

    @staticmethod
    def transition(learner, seed):
        from src.vec.learning import HybridTransition

        random = np.random.default_rng(seed)
        observations = {}
        next_observations = {}
        task_features = {}
        next_task_features = {}
        task_masks = {}
        next_task_masks = {}
        categorical_masks = {}
        next_categorical_masks = {}
        for spec in learner.role_specs:
            observations[spec.name] = random.normal(
                size=(spec.count, spec.observation_dim)
            ).astype(np.float32)
            next_observations[spec.name] = random.normal(
                size=(spec.count, spec.observation_dim)
            ).astype(np.float32)
            task_features[spec.name] = random.normal(
                size=(spec.count, spec.max_tasks, spec.task_feature_dim)
            ).astype(np.float32)
            next_task_features[spec.name] = random.normal(
                size=(spec.count, spec.max_tasks, spec.task_feature_dim)
            ).astype(np.float32)
            mask = np.zeros((spec.count, spec.max_tasks), dtype=np.bool_)
            mask[:, :2] = True
            task_masks[spec.name] = mask
            next_task_masks[spec.name] = mask.copy()
            categorical_masks[spec.name] = np.ones(
                (spec.count, spec.categorical_dim), dtype=np.bool_
            )
            next_categorical_masks[spec.name] = categorical_masks[
                spec.name
            ].copy()
        actions = learner.act(
            observations,
            task_features,
            task_masks,
            categorical_masks,
            deterministic=True,
        )
        next_actions = learner.act(
            next_observations,
            next_task_features,
            next_task_masks,
            next_categorical_masks,
            deterministic=True,
        )
        return HybridTransition(
            state=random.normal(size=learner.config.state_dim).astype(np.float32),
            next_state=random.normal(size=learner.config.state_dim).astype(
                np.float32
            ),
            observations=observations,
            next_observations=next_observations,
            task_features=task_features,
            next_task_features=next_task_features,
            task_masks=task_masks,
            next_task_masks=next_task_masks,
            categorical_masks=categorical_masks,
            next_categorical_masks=next_categorical_masks,
            proposed_actions=actions,
            executed_actions=next_actions,
            rewards={spec.name: float(seed) / 10.0 for spec in learner.role_specs},
            done=False,
        )

    def test_masked_hybrid_actor_enforces_destinations_and_task_padding(self):
        from src.vec.learning import HybridRoleActor

        spec = self.RoleSpec(
            "vehicle", 1, 3, 2, 3, categorical_dim=4, has_cpu=True
        )
        actor = HybridRoleActor(spec, hidden_dim=8)
        action = actor(
            torch.zeros((1, 1, 3)),
            torch.ones((1, 1, 3, 2)),
            torch.tensor([[[True, False, True]]]),
            torch.tensor([[[False, False, True, False]]]),
            deterministic=True,
        )

        self.assertEqual(tuple(action.shape), (1, 1, spec.action_dim))
        self.assertEqual(action[0, 0, :4].tolist(), [0.0, 0.0, 1.0, 0.0])
        shares = action[0, 0, -3:]
        self.assertEqual(float(shares[1].detach()), 0.0)
        self.assertAlmostEqual(float(shares.sum().detach()), 1.0, places=6)

    def test_smoke_update_checkpoint_round_trip_and_gradient_diagnostics(self):
        from src.vec.learning import HybridMADDPG

        learner = HybridMADDPG(self.config())
        learner.begin_meta_task("scenario-a")
        for seed in (1, 2, 3):
            learner.remember(self.transition(learner, seed))
        diagnostics = learner.update()
        self.assertEqual(learner.update_count, 1)
        self.assertTrue(
            all(np.isfinite(value) for value in diagnostics.actor_gradient_norm.values())
        )
        sample = self.transition(learner, 9)
        expected = learner.act(
            sample.observations,
            sample.task_features,
            sample.task_masks,
            sample.categorical_masks,
            deterministic=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase5.pt"
            learner.save(path)
            restored = HybridMADDPG(self.config())
            restored.load(path)

        actual = restored.act(
            sample.observations,
            sample.task_features,
            sample.task_masks,
            sample.categorical_masks,
            deterministic=True,
        )
        self.assertEqual(restored.update_count, learner.update_count)
        self.assertEqual(restored.current_task_id, "scenario-a")
        self.assertEqual(restored.replay.size("scenario-a"), 3)
        for role in expected:
            np.testing.assert_array_equal(actual[role], expected[role])
        for name, parameter in learner.actors.state_dict().items():
            self.assertTrue(torch.equal(parameter, restored.actors.state_dict()[name]))

    def test_new_meta_task_clears_replay_and_optimizer_moments(self):
        from src.vec.learning import HybridMADDPG

        learner = HybridMADDPG(self.config())
        learner.begin_meta_task("first")
        for seed in (1, 2):
            learner.remember(self.transition(learner, seed))
        learner.update()
        self.assertTrue(
            any(optimizer.state for optimizer in learner.actor_optimizers.values())
        )

        learner.begin_meta_task("second")

        self.assertEqual(learner.replay.task_ids(), ("second",))
        self.assertTrue(
            all(not optimizer.state for optimizer in learner.actor_optimizers.values())
        )
        for name, value in learner.actors.state_dict().items():
            self.assertTrue(torch.equal(value, learner.target_actors.state_dict()[name]))

    def test_reptile_updates_actors_and_critics_then_synchronizes_targets(self):
        from src.vec.learning import HybridMADDPG, ReptileMetaLearner

        learner = HybridMADDPG(self.config())
        original = learner.online_state_dict()
        adapted_a = learner.fork()
        adapted_b = learner.fork()
        with torch.no_grad():
            for adapted, shift in ((adapted_a, 1.0), (adapted_b, 3.0)):
                for module_name in ("actors", "critics"):
                    for parameter in getattr(adapted, module_name).parameters():
                        parameter.add_(shift)
        ReptileMetaLearner(learner, outer_step_size=0.5).outer_update(
            (adapted_a, adapted_b)
        )

        for module_name, target_name in (
            ("actors", "target_actors"),
            ("critics", "target_critics"),
        ):
            online = getattr(learner, module_name).state_dict()
            target = getattr(learner, target_name).state_dict()
            for name, value in online.items():
                self.assertTrue(torch.equal(value, target[name]))
                self.assertTrue(
                    torch.allclose(value, original[module_name][name] + 1.0)
                )

    def test_toy_tasks_adapt_faster_from_reptile_initialization(self):
        from src.vec.learning import HybridMADDPG, ReptileMetaLearner

        meta = HybridMADDPG(self.config(seed=11))
        initial = meta.online_state_dict()
        task_learners = (meta.fork(), meta.fork())
        with torch.no_grad():
            for task, displacement in zip(task_learners, (0.8, 1.2)):
                for parameter in task.actors.parameters():
                    parameter.add_(displacement)
                for parameter in task.critics.parameters():
                    parameter.add_(displacement)
        ReptileMetaLearner(meta, outer_step_size=1.0).outer_update(task_learners)

        target = [parameter.detach().clone() for parameter in meta.actors.parameters()]
        random_start = HybridMADDPG(self.config(seed=11))
        random_start.load_online_state_dict(initial)

        def one_matched_step(learner):
            optimizer = torch.optim.SGD(learner.actors.parameters(), lr=0.1)
            optimizer.zero_grad()
            loss = sum(
                (parameter - goal).pow(2).mean()
                for parameter, goal in zip(
                    learner.actors.parameters(), target, strict=True
                )
            )
            loss.backward()
            optimizer.step()
            return float(
                sum(
                    (parameter - goal).pow(2).mean()
                    for parameter, goal in zip(
                        learner.actors.parameters(), target, strict=True
                    )
                ).detach()
            )

        self.assertLess(one_matched_step(meta), one_matched_step(random_start))

    def test_adapter_emits_step_action_and_online_adaptation_keeps_env_state(self):
        from src.vec import VecQueueEnv
        from src.vec.learning import BoundedOnlineAdapter, VecPolicyAdapter

        environment = VecQueueEnv()
        task_id = environment.submit_task(
            0,
            input_bits=1.0e6,
            cpu_cycles=1.0e9,
            deadline=10.0,
            partitionable=True,
        )
        adapter = VecPolicyAdapter(environment, max_tasks=3)
        encoded = adapter.encode()
        actions = {
            "vehicle": np.asarray([[0.0, 0.0, 0.0, 1.0, 0.25]], dtype=np.float32),
            "rsu": np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            "uav": np.asarray(
                [[0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32
            ),
        }
        step_action = adapter.decode(encoded, actions)
        self.assertEqual(step_action.dispatches[0].task_id, task_id)
        self.assertAlmostEqual(step_action.dispatches[0].uav_fraction, 0.25)
        self.assertAlmostEqual(step_action.uav_cpu_frequencies_hz[0], 2.0e9)

        class ReadyLearner:
            calls = 0

            def ready(self):
                return True

            def update(self):
                self.calls += 1
                return self.calls

        stub = ReadyLearner()
        online = BoundedOnlineAdapter(stub, max_updates_per_change=2)
        before = environment.control_observation()
        online.declare_change("traffic-surge", environment)
        report = online.adapt(9)
        after = environment.control_observation()
        self.assertEqual(report.completed_updates, 2)
        self.assertEqual(report.remaining_budget, 0)
        self.assertEqual(before, after)
        self.assertEqual(environment.slot, 0)

        result = environment.step(step_action)
        encoded_after = adapter.encode(
            result.observation_after, result.masks_after
        )
        transition = adapter.transition(encoded, encoded_after, actions, result)
        self.assertEqual(transition.state.shape, encoded.state.shape)
        self.assertEqual(transition.done, result.terminated or result.truncated)
        self.assertAlmostEqual(
            transition.rewards["vehicle"], result.lyapunov.reward
        )

    def test_checkpoint_controller_runs_through_phase6_common_evaluator(self):
        from src.vec.evaluation import (
            EvaluationMethod,
            ScenarioSpec,
            evaluate_scenario,
            instantiate_scenario,
        )
        from src.vec.learning import (
            HybridMADDPG,
            HybridMADDPGConfig,
            HybridPolicyController,
            VecPolicyAdapter,
        )

        scenario = instantiate_scenario(
            ScenarioSpec(
                "test-learned-smoke",
                "test",
                "low_load",
                777,
                horizon_slots=2,
                drain_slots=1,
            )
        )
        environment = scenario.environment()
        adapter = VecPolicyAdapter(environment, max_tasks=4)
        learner = HybridMADDPG(
            HybridMADDPGConfig(
                state_dim=len(
                    environment.control_observation().global_critic_state
                ),
                role_specs=adapter.role_specs(),
                actor_hidden_dim=8,
                critic_hidden_dim=12,
                batch_size=1,
                replay_capacity=8,
                seed=19,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            learner.save(checkpoint)

            def factory(instance, run_environment):
                return HybridPolicyController.from_checkpoint(
                    "hybrid_maddpg", checkpoint, run_environment
                )

            run = evaluate_scenario(
                scenario,
                EvaluationMethod("hybrid_maddpg", "learned", factory),
            )

        self.assertEqual(run.method, "hybrid_maddpg")
        self.assertEqual(run.trace_digest, scenario.trace_digest)
        self.assertGreater(len(run.traces), 0)


if __name__ == "__main__":
    unittest.main()
