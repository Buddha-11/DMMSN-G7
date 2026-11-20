"""Tests for fair non-learning baselines and their shared execution path."""

import unittest

from src.vec.baselines import (
    GeneticDriftController,
    LocalOnlyController,
    NearestFeasibleEdgeController,
    QueueEnergyGreedyController,
    RandomFeasibleController,
    execute_controller_step,
    run_controller_episode,
)
from src.vec.env import VecEnvConfig, VecQueueEnv
from src.vec.tasks import DestinationKind


class BaselineTests(unittest.TestCase):
    @staticmethod
    def environment(**overrides):
        values = dict(
            vehicle_cpu_hz=(1.0,),
            rsu_cpu_hz=(10.0, 10.0),
            uav_cpu_hz=(2.0,),
            rsu_uplink_bps=(100.0, 100.0),
            uav_uplink_bps=(100.0,),
            vehicle_static_positions=((0.0, 0.0, 0.0),),
            rsu_positions=((100.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
            uav_positions=((5.0, 0.0, 100.0),),
        )
        values.update(overrides)
        env = VecQueueEnv(VecEnvConfig(**values))
        env.submit_task(
            0,
            input_bits=10.0,
            cpu_cycles=10.0,
            deadline=20.0,
            partitionable=True,
        )
        return env

    def test_local_only_routes_oldest_ingress_task_to_source_cpu(self):
        env = self.environment()
        decision = LocalOnlyController().select_action(env)

        self.assertEqual(len(decision.action.dispatches), 1)
        dispatch = decision.action.dispatches[0]
        self.assertEqual(dispatch.destination.kind, DestinationKind.LOCAL)
        self.assertEqual(dispatch.destination.node_id, 0)
        self.assertEqual(decision.diagnostics.controller, "local_only")

    def test_nearest_feasible_edge_uses_deterministic_distance_ties(self):
        env = self.environment()
        decision = NearestFeasibleEdgeController().select_action(env)

        dispatch = decision.action.dispatches[0]
        self.assertEqual(dispatch.destination.kind, DestinationKind.RSU)
        self.assertEqual(dispatch.destination.node_id, 1)
        self.assertGreater(decision.diagnostics.candidates_evaluated, 0)

    def test_queue_energy_greedy_prefers_faster_uncongested_edge(self):
        env = self.environment(
            uav_cpu_hz=(),
            uav_uplink_bps=(),
            uav_positions=(),
            rsu_uplink_bps=(10.0, 100.0),
        )
        decision = QueueEnergyGreedyController().select_action(env)

        dispatch = decision.action.dispatches[0]
        self.assertEqual(dispatch.destination.kind, DestinationKind.RSU)
        self.assertEqual(dispatch.destination.node_id, 1)

    def test_random_feasible_is_seeded_and_never_selects_masked_option(self):
        first_env = self.environment()
        second_env = self.environment()
        first = RandomFeasibleController(seed=17, allow_hold=False)
        second = RandomFeasibleController(seed=17, allow_hold=False)

        first_decision = first.select_action(first_env)
        second_decision = second.select_action(second_env)

        self.assertEqual(first_decision.action, second_decision.action)
        dispatch = first_decision.action.dispatches[0]
        mask = first_env.destination_masks()[0]
        if dispatch.destination.kind is DestinationKind.RSU:
            self.assertTrue(mask.rsu[dispatch.destination.node_id])
        elif dispatch.destination.kind is DestinationKind.UAV:
            self.assertTrue(mask.uav[dispatch.destination.node_id])
        else:
            self.assertEqual(dispatch.destination.node_id, 0)

    def test_genetic_drift_is_deterministic_and_uses_exact_budget(self):
        first_env = self.environment(uav_cpu_hz=(), uav_uplink_bps=(), uav_positions=())
        second_env = self.environment(uav_cpu_hz=(), uav_uplink_bps=(), uav_positions=())
        first = GeneticDriftController(
            evaluation_budget=11, population_size=4, seed=9
        )
        second = GeneticDriftController(
            evaluation_budget=11, population_size=4, seed=9
        )

        first_decision = first.select_action(first_env)
        second_decision = second.select_action(second_env)

        self.assertEqual(first_decision.action, second_decision.action)
        self.assertEqual(first_decision.diagnostics.candidates_evaluated, 11)
        self.assertEqual(first_decision.diagnostics.evaluation_budget, 11)
        self.assertGreater(first_decision.diagnostics.generations, 0)
        self.assertIsNotNone(first_decision.diagnostics.best_score)

    def test_different_controllers_share_identical_execution_path(self):
        config = dict(
            rsu_cpu_hz=(),
            uav_cpu_hz=(),
            rsu_uplink_bps=(),
            uav_uplink_bps=(),
            rsu_positions=(),
            uav_positions=(),
        )
        local_env = self.environment(**config)
        random_env = self.environment(**config)
        local = LocalOnlyController()
        random = RandomFeasibleController(seed=3, allow_hold=False)

        local_step = execute_controller_step(local_env, local)
        random_step = execute_controller_step(random_env, random)

        self.assertEqual(local_step.decision.action, random_step.decision.action)
        self.assertEqual(
            local_step.result.action_trace,
            random_step.result.action_trace,
        )
        self.assertEqual(local_step.result.queue_after, random_step.result.queue_after)
        self.assertEqual(
            local_step.result.lyapunov.drift_plus_penalty,
            random_step.result.lyapunov.drift_plus_penalty,
        )
        self.assertGreaterEqual(local_step.decision.diagnostics.elapsed_seconds, 0.0)
        self.assertGreaterEqual(random_step.decision.diagnostics.elapsed_seconds, 0.0)

    def test_common_rollout_collects_identical_metrics_schema(self):
        config = dict(
            rsu_cpu_hz=(),
            uav_cpu_hz=(),
            rsu_uplink_bps=(),
            uav_uplink_bps=(),
            rsu_positions=(),
            uav_positions=(),
            max_episode_slots=2,
        )
        local_run = run_controller_episode(
            self.environment(**config), LocalOnlyController(), 5
        )
        random_run = run_controller_episode(
            self.environment(**config),
            RandomFeasibleController(seed=5, allow_hold=False),
            5,
        )

        self.assertEqual(local_run.metrics.slots, 2)
        self.assertEqual(local_run.metrics.slots, random_run.metrics.slots)
        self.assertEqual(
            local_run.metrics.completed_tasks,
            random_run.metrics.completed_tasks,
        )
        self.assertEqual(
            local_run.metrics.peak_backlog_cycles,
            random_run.metrics.peak_backlog_cycles,
        )


if __name__ == "__main__":
    unittest.main()
