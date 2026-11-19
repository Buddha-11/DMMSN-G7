"""Analytical tests for the Phase-3 objective and control interface."""

import unittest

from src.vec.energy import RotaryWingEnergyConfig
from src.vec.env import (
    Destination,
    DispatchDecision,
    ScheduledHarvest,
    StepAction,
    VecEnvConfig,
    VecQueueEnv,
)
from src.vec.lyapunov import LyapunovConfig, LyapunovSlotInput, LyapunovTracker


def slot_input(**overrides):
    values = dict(
        queue_before={"a": 20.0, "b": 0.0},
        queue_after={"a": 15.0, "b": 5.0},
        queue_arrivals={"b": 5.0},
        queue_departures={"a": 5.0},
        uav_energy_j=(4.0,),
        harvest_j=(1.0,),
        system_energy_j=5.0,
        unfinished_tasks=3,
        new_deadline_misses=1,
        slot_duration_s=1.0,
        external_arrival_cycles=0.0,
        compute_service_cycles=0.0,
    )
    values.update(overrides)
    return LyapunovSlotInput(**values)


class LyapunovTests(unittest.TestCase):
    def config(self):
        return LyapunovConfig(
            queue_reference_cycles=10.0,
            energy_queue_reference_j=10.0,
            energy_cost_reference_j=10.0,
            latency_reference_s=2.0,
            tradeoff_v=2.0,
            latency_weight=1.0,
            energy_weight=2.0,
            deadline_weight=3.0,
            net_energy_target_j=(1.0,),
            expected_harvest_j=(7.0,),
        )

    def test_symbolic_two_queue_score_matches_hand_calculation(self):
        tracker = LyapunovTracker(1, self.config())
        tracker.virtual_energy_j[0] = 10.0

        score = tracker.record(slot_input())

        self.assertAlmostEqual(score.latency_cost, 1.5)
        self.assertAlmostEqual(score.energy_cost, 1.0)
        self.assertAlmostEqual(score.deadline_cost, 3.0)
        self.assertAlmostEqual(score.immediate_cost, 5.5)
        self.assertAlmostEqual(score.physical_queue_term, -1.0)
        self.assertAlmostEqual(score.virtual_energy_term, 0.2)
        self.assertAlmostEqual(score.drift_plus_penalty, 10.2)
        self.assertAlmostEqual(score.reward, -10.2)
        self.assertEqual(score.virtual_energy_after_j, (12.0,))
        self.assertAlmostEqual(score.flow_balance_residual_cycles, 0.0)

    def test_preview_uses_expected_harvest_without_mutating_virtual_queue(self):
        tracker = LyapunovTracker(1, self.config())
        tracker.virtual_energy_j[0] = 10.0

        preview = tracker.preview(slot_input(harvest_j=(999.0,)))

        self.assertTrue(preview.used_expected_harvest)
        self.assertEqual(preview.net_energy_increment_j, (-4.0,))
        self.assertEqual(preview.virtual_energy_after_j, (6.0,))
        self.assertEqual(tracker.virtual_energy_j, [10.0])

    def test_unbalanced_internal_transfer_is_rejected(self):
        tracker = LyapunovTracker(1, self.config())
        with self.assertRaisesRegex(AssertionError, "transfer ledger"):
            tracker.record(
                slot_input(
                    queue_arrivals={"b": 4.0},
                    queue_departures={"a": 5.0},
                )
            )

    def test_large_balanced_transfer_tolerates_floating_point_roundoff(self):
        tracker = LyapunovTracker(0)
        amount = 41_056_947_442.52153
        tracker.record(
            LyapunovSlotInput(
                queue_before={"source": amount, "destination": 0.0},
                queue_after={"source": 0.0, "destination": amount},
                queue_arrivals={"destination": amount},
                queue_departures={"source": amount + 1.52587890625e-5},
                uav_energy_j=(),
                harvest_j=(),
                system_energy_j=0.0,
                unfinished_tasks=1,
                new_deadline_misses=0,
                slot_duration_s=1.0,
                external_arrival_cycles=0.0,
                compute_service_cycles=0.0,
            )
        )

    def test_large_external_and_compute_flows_tolerate_cancellation_residue(self):
        tracker = LyapunovTracker(0)
        amount = 4_000_000_000.0
        score = tracker.record(
            LyapunovSlotInput(
                queue_before={"ingress": 0.0, "compute": amount},
                queue_after={"ingress": amount, "compute": 0.0},
                queue_arrivals={"ingress": amount},
                queue_departures={"compute": amount + 1.0728836059570312e-5},
                uav_energy_j=(),
                harvest_j=(),
                system_energy_j=0.0,
                unfinished_tasks=1,
                new_deadline_misses=0,
                slot_duration_s=1.0,
                external_arrival_cycles=amount,
                compute_service_cycles=amount,
            )
        )
        self.assertLess(abs(score.flow_balance_residual_cycles), 2.0e-5)


class ControlInterfaceTests(unittest.TestCase):
    @staticmethod
    def electronics_only():
        return RotaryWingEnergyConfig(
            profile_power_w=0.0,
            induced_power_w=0.0,
            parasitic_coefficient=0.0,
            ascent_power_per_mps_w=0.0,
            descent_power_per_mps_w=0.0,
            vertical_quadratic_w_per_mps2=0.0,
            compute_kappa=0.0,
            receive_power_w=0.0,
            backhaul_tx_power_w=0.0,
            control_power_w=1.0,
        )

    def test_virtual_energy_uses_realized_end_of_slot_harvest(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(0.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(0.0,),
            uav_energy=self.electronics_only(),
            lyapunov=LyapunovConfig(
                energy_queue_reference_j=1.0,
                expected_harvest_j=(100.0,),
            ),
        )
        env = VecQueueEnv(
            config,
            harvest_trace=(ScheduledHarvest(0, 0, 2.0),),
        )

        first = env.step()
        second = env.step()

        self.assertEqual(first.lyapunov.virtual_energy_after_j, (0.0,))
        self.assertFalse(first.lyapunov.used_expected_harvest)
        self.assertEqual(second.lyapunov.virtual_energy_after_j, (1.0,))
        self.assertEqual(
            second.observation_after.uavs[0].features["virtual_energy_q"], 1.0
        )

    def test_observations_masks_action_trace_and_truncation_are_exposed(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(2.0,),
            rsu_cpu_hz=(2.0,),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(2.0,),
            uav_uplink_bps=(1.0,),
            max_episode_slots=1,
            lyapunov=LyapunovConfig(queue_reference_cycles=10.0),
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(
            0, input_bits=2.0, cpu_cycles=2.0, deadline=10.0
        )
        action = StepAction(
            dispatches=(
                DispatchDecision(0, task_id, Destination.local(0)),
            ),
            uav_cpu_frequencies_hz=(0.5,),
        )

        result = env.step(action)

        self.assertAlmostEqual(
            result.observation_before.vehicles[0].features["ingress_q"], 0.2
        )
        self.assertEqual(result.masks_before[0].ingress_task_ids, (task_id,))
        self.assertTrue(result.masks_before[0].local)
        self.assertEqual(result.action_trace.dispatches[0].task_id, task_id)
        self.assertEqual(result.action_trace.requested_uav_cpu_hz, (0.5,))
        self.assertEqual(result.action_trace.executed_uav_cpu_hz, (0.5,))
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertEqual(
            len(result.observation_after.global_critic_state),
            len(result.observation_after.global_feature_names),
        )
        with self.assertRaisesRegex(RuntimeError, "episode is done"):
            env.step()


if __name__ == "__main__":
    unittest.main()
