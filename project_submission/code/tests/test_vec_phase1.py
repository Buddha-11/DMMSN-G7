"""Analytically checkable Phase-1 queue-network tests."""

import unittest

from src.vec import (
    Destination,
    DispatchDecision,
    StepAction,
    TaskState,
    VecEnvConfig,
    VecQueueEnv,
)
from src.vec.env import ScheduledArrival
from src.vec.queues import QueueStage


class VecPhaseOneTests(unittest.TestCase):
    def test_local_task_persists_until_cpu_service_finishes(self):
        env = VecQueueEnv(
            VecEnvConfig(
                vehicle_cpu_hz=(50.0,),
                rsu_cpu_hz=(),
                uav_cpu_hz=(),
                rsu_uplink_bps=(),
                uav_uplink_bps=(),
            )
        )
        task_id = env.submit_task(0, input_bits=10.0, cpu_cycles=100.0, deadline=10)
        first = env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, task_id, Destination.local(0)),
                )
            )
        )

        self.assertEqual(first.completed_task_ids, ())
        self.assertEqual(env.tasks[task_id].state, TaskState.PROCESSING)
        self.assertAlmostEqual(env.total_backlog_cycles, 50.0)

        second = env.step()
        self.assertEqual(second.completed_task_ids, (task_id,))
        self.assertEqual(env.tasks[task_id].completion_time, 2.0)
        self.assertAlmostEqual(env.total_backlog_cycles, 0.0)

    def test_completed_upload_cannot_compute_in_the_same_slot(self):
        env = VecQueueEnv(
            VecEnvConfig(
                vehicle_cpu_hz=(0.0,),
                rsu_cpu_hz=(100.0,),
                uav_cpu_hz=(),
                rsu_uplink_bps=(10.0,),
                uav_uplink_bps=(),
            )
        )
        task_id = env.submit_task(0, input_bits=10.0, cpu_cycles=100.0, deadline=10)
        first = env.step(
            StepAction(
                dispatches=(DispatchDecision(0, task_id, Destination.rsu(0)),)
            )
        )

        self.assertEqual(first.completed_task_ids, ())
        self.assertAlmostEqual(
            env.get_queue(QueueStage.COMPUTE_RSU, 0).equivalent_backlog, 100.0
        )
        second = env.step()
        self.assertEqual(second.completed_task_ids, (task_id,))
        self.assertEqual(env.tasks[task_id].completion_time, 2.0)

    def test_partial_upload_and_staging_conserve_equivalent_work(self):
        env = VecQueueEnv(
            VecEnvConfig(
                vehicle_cpu_hz=(0.0,),
                rsu_cpu_hz=(100.0,),
                uav_cpu_hz=(),
                rsu_uplink_bps=(4.0,),
                uav_uplink_bps=(),
            )
        )
        task_id = env.submit_task(0, input_bits=10.0, cpu_cycles=100.0, deadline=10)
        result = env.step(
            StepAction(
                dispatches=(DispatchDecision(0, task_id, Destination.rsu(0)),)
            )
        )

        self.assertAlmostEqual(
            env.get_queue(QueueStage.UPLOAD_RSU, 0).amount_for(task_id), 6.0
        )
        self.assertAlmostEqual(
            env.get_queue(QueueStage.RECEIVE_RSU, 0).amount_for(task_id), 4.0
        )
        self.assertAlmostEqual(env.total_backlog_cycles, 100.0)
        self.assertAlmostEqual(result.conservation_residual, 0.0)

    def test_deadline_miss_is_one_event_and_does_not_drop_work(self):
        env = VecQueueEnv(
            VecEnvConfig(
                vehicle_cpu_hz=(20.0,),
                rsu_cpu_hz=(),
                uav_cpu_hz=(),
                rsu_uplink_bps=(),
                uav_uplink_bps=(),
            )
        )
        task_id = env.submit_task(0, input_bits=10.0, cpu_cycles=100.0, deadline=1.0)
        env.step(
            StepAction(
                dispatches=(DispatchDecision(0, task_id, Destination.local(0)),)
            )
        )
        second = env.step()
        third = env.step()

        self.assertEqual(second.new_deadline_miss_ids, (task_id,))
        self.assertEqual(third.new_deadline_miss_ids, ())
        self.assertFalse(env.tasks[task_id].is_complete)
        self.assertAlmostEqual(env.total_backlog_cycles, 40.0)

        env.step()
        final = env.step()
        self.assertEqual(final.completed_task_ids, (task_id,))
        self.assertTrue(env.tasks[task_id].deadline_missed)

    def test_trace_arrivals_enter_only_at_the_declared_boundary(self):
        trace = (
            ScheduledArrival(0, 0, 10.0, 100.0, 5.0, task_id=7),
            ScheduledArrival(1, 0, 20.0, 200.0, 5.0, task_id=8),
        )
        env = VecQueueEnv(
            VecEnvConfig(
                vehicle_cpu_hz=(0.0,),
                rsu_cpu_hz=(),
                uav_cpu_hz=(),
                rsu_uplink_bps=(),
                uav_uplink_bps=(),
            ),
            arrival_trace=trace,
        )
        self.assertEqual(set(env.tasks), {7})
        result = env.step()
        self.assertEqual(result.new_arrival_ids, (8,))
        self.assertEqual(set(env.tasks), {7, 8})
        self.assertAlmostEqual(env.total_backlog_cycles, 300.0)

    def test_weighted_processor_sharing_redistributes_capacity(self):
        env = VecQueueEnv(
            VecEnvConfig(
                vehicle_cpu_hz=(40.0,),
                rsu_cpu_hz=(),
                uav_cpu_hz=(),
                rsu_uplink_bps=(),
                uav_uplink_bps=(),
                max_dispatches_per_vehicle=2,
            )
        )
        first_id = env.submit_task(0, 10.0, 20.0, 10.0)
        second_id = env.submit_task(0, 10.0, 100.0, 10.0)
        key = "compute_local:0"
        env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, first_id, Destination.local(0)),
                    DispatchDecision(0, second_id, Destination.local(0)),
                ),
                cpu_weights={key: {first_id: 1.0, second_id: 3.0}},
            )
        )

        queue = env.get_queue(QueueStage.COMPUTE_LOCAL, 0)
        self.assertAlmostEqual(queue.amount_for(first_id), 10.0)
        self.assertAlmostEqual(queue.amount_for(second_id), 70.0)


if __name__ == "__main__":
    unittest.main()
