"""Tests for fixed UAV/cloud splitting, backhaul, merge, and completion."""

import unittest

from src.vec import (
    Destination,
    DispatchDecision,
    StepAction,
    TaskState,
    VecEnvConfig,
    VecQueueEnv,
)
from src.vec.queues import QueueStage


class CloudSplitTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = dict(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(100.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(10.0,),
            uav_backhaul_bps=10.0,
            cloud_cpu_hz=100.0,
        )
        values.update(overrides)
        return VecEnvConfig(**values)

    def dispatch(self, env, fraction):
        task_id = env.submit_task(
            0,
            input_bits=10.0,
            cpu_cycles=100.0,
            deadline=100.0,
            partitionable=True,
        )
        first = env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(
                        0, task_id, Destination.uav(0), uav_fraction=fraction
                    ),
                )
            )
        )
        return task_id, first

    def test_all_uav_endpoint_skips_backhaul_and_cloud(self):
        env = VecQueueEnv(self.config())
        task_id, first = self.dispatch(env, 1.0)
        self.assertEqual(first.completed_task_ids, ())
        self.assertAlmostEqual(
            env.get_queue(QueueStage.COMPUTE_UAV, 0).native_backlog, 100.0
        )
        self.assertEqual(
            env.get_queue(QueueStage.BACKHAUL_UAV, 0).native_backlog, 0.0
        )

        second = env.step()
        self.assertEqual(second.completed_task_ids, (task_id,))
        self.assertEqual(env.tasks[task_id].completed_branches, {"uav"})

    def test_all_cloud_endpoint_obeys_two_stage_causality(self):
        env = VecQueueEnv(self.config())
        task_id, _ = self.dispatch(env, 0.0)
        self.assertAlmostEqual(
            env.get_queue(QueueStage.BACKHAUL_UAV, 0).native_backlog, 10.0
        )
        self.assertEqual(env.get_queue(QueueStage.COMPUTE_UAV, 0).native_backlog, 0.0)

        second = env.step()
        self.assertEqual(second.completed_task_ids, ())
        self.assertAlmostEqual(
            env.get_queue(QueueStage.COMPUTE_CLOUD, 0).native_backlog, 100.0
        )

        third = env.step()
        self.assertEqual(third.completed_task_ids, (task_id,))
        self.assertEqual(env.tasks[task_id].completed_branches, {"cloud"})

    def test_split_task_waits_for_slower_branch(self):
        env = VecQueueEnv(self.config(uav_cpu_hz=(10.0,)))
        task_id, _ = self.dispatch(env, 0.5)

        env.step()
        third = env.step()
        self.assertEqual(third.completed_task_ids, ())
        self.assertEqual(env.tasks[task_id].completed_branches, {"cloud"})
        self.assertEqual(env.tasks[task_id].state, TaskState.PROCESSING)

        env.step()
        env.step()
        sixth = env.step()
        self.assertEqual(sixth.completed_task_ids, (task_id,))
        self.assertEqual(env.tasks[task_id].completed_branches, {"uav", "cloud"})
        self.assertAlmostEqual(env.total_backlog_cycles, 0.0)

    def test_merge_delay_preserves_exact_completion_timestamp(self):
        env = VecQueueEnv(self.config(merge_delay_s=1.5))
        task_id, _ = self.dispatch(env, 1.0)
        branch_done = env.step()
        self.assertEqual(branch_done.completed_task_ids, ())
        self.assertEqual(env.tasks[task_id].merge_ready_time, 3.5)

        self.assertEqual(env.step().completed_task_ids, ())
        merged = env.step()
        self.assertEqual(merged.completed_task_ids, (task_id,))
        self.assertEqual(env.tasks[task_id].completion_time, 3.5)

    def test_indivisible_task_cannot_be_cloud_split(self):
        env = VecQueueEnv(self.config())
        task_id = env.submit_task(0, 10.0, 100.0, 100.0, partitionable=False)
        with self.assertRaisesRegex(ValueError, "indivisible"):
            env.step(
                StepAction(
                    dispatches=(
                        DispatchDecision(
                            0, task_id, Destination.uav(0), uav_fraction=0.5
                        ),
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
