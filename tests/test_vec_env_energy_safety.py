"""Integration tests for Phase-2 energy and flight behavior."""

import unittest

from src.vec.energy import (
    BatteryConfig,
    RotaryWingEnergyConfig,
    SystemEnergyConfig,
)
from src.vec.env import (
    Destination,
    DispatchDecision,
    ScheduledHarvest,
    StepAction,
    UAVMode,
    VecEnvConfig,
    VecQueueEnv,
)
from src.vec.queues import QueueStage
from src.vec.safety import FlightLimits, MotionCommand


def electronics_only(**overrides):
    values = dict(
        profile_power_w=0.0,
        induced_power_w=0.0,
        parasitic_coefficient=0.0,
        ascent_power_per_mps_w=0.0,
        descent_power_per_mps_w=0.0,
        vertical_quadratic_w_per_mps2=0.0,
        compute_kappa=0.0,
        receive_power_w=0.0,
        backhaul_tx_power_w=0.0,
        control_power_w=0.0,
    )
    values.update(overrides)
    return RotaryWingEnergyConfig(**values)


class EnvironmentEnergyTests(unittest.TestCase):
    @staticmethod
    def system_energy():
        return SystemEnergyConfig(
            vehicle_compute_kappa=0.1,
            rsu_compute_kappa=0.1,
            cloud_compute_kappa=0.1,
            vehicle_tx_power_w=0.5,
            rsu_receive_power_w=0.25,
            cloud_receive_power_w=0.2,
        )

    def test_non_uav_energy_is_charged_to_its_physical_owner(self):
        local_config = VecEnvConfig(
            vehicle_cpu_hz=(2.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(),
            rsu_uplink_bps=(),
            uav_uplink_bps=(),
            system_energy=self.system_energy(),
        )
        local = VecQueueEnv(local_config)
        local_task = local.submit_task(
            0, input_bits=1.0, cpu_cycles=2.0, deadline=10.0
        )
        local_result = local.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, local_task, Destination.local(0)),
                )
            )
        )
        self.assertAlmostEqual(local_result.energy.vehicle_compute_j[0], 0.8)
        self.assertAlmostEqual(local_result.energy.total_j, 0.8)

        rsu_config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(2.0,),
            uav_cpu_hz=(),
            rsu_uplink_bps=(2.0,),
            uav_uplink_bps=(),
            system_energy=self.system_energy(),
        )
        rsu = VecQueueEnv(rsu_config)
        rsu_task = rsu.submit_task(
            0, input_bits=2.0, cpu_cycles=2.0, deadline=10.0
        )
        upload = rsu.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, rsu_task, Destination.rsu(0)),
                )
            )
        )
        compute = rsu.step()
        self.assertAlmostEqual(upload.energy.vehicle_uplink_j[0], 0.5)
        self.assertAlmostEqual(upload.energy.rsu_reception_j[0], 0.25)
        self.assertAlmostEqual(upload.energy.total_j, 0.75)
        self.assertAlmostEqual(compute.energy.rsu_compute_j[0], 0.8)
        self.assertAlmostEqual(compute.energy.total_j, 0.8)

    def test_end_of_slot_harvest_cannot_fund_current_reception(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(10.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(10.0,),
            uav_batteries=(BatteryConfig(10.0, 1.0),),
            uav_energy=electronics_only(receive_power_w=2.0),
        )
        env = VecQueueEnv(
            config,
            harvest_trace=(ScheduledHarvest(slot=0, uav_id=0, energy_j=10.0),),
        )
        task_id = env.submit_task(0, input_bits=10.0, cpu_cycles=10.0, deadline=10.0)

        result = env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, task_id, Destination.uav(0)),
                )
            )
        )

        self.assertAlmostEqual(
            env.get_queue(QueueStage.UPLOAD_UAV, 0).native_backlog, 5.0
        )
        self.assertAlmostEqual(result.uav_energy_j[0], 1.0)
        self.assertAlmostEqual(result.battery_transitions[0].charge_before_j, 1.0)
        self.assertAlmostEqual(result.battery_transitions[0].charge_after_j, 10.0)
        self.assertAlmostEqual(result.battery_transitions[0].overflow_j, 0.0)

    def test_compute_energy_is_charged_from_actual_uav_cycles_once(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(2.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0,),
            uav_batteries=(BatteryConfig(10.0, 10.0),),
            uav_energy=electronics_only(compute_kappa=0.1),
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(0, input_bits=1.0, cpu_cycles=4.0, deadline=10.0)
        env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, task_id, Destination.uav(0)),
                )
            )
        )

        result = env.step(StepAction(uav_cpu_frequencies_hz=(2.0,)))

        self.assertAlmostEqual(result.compute_service_cycles, 2.0)
        self.assertAlmostEqual(result.uav_energy_j[0], 0.8)
        self.assertAlmostEqual(result.battery_transitions[0].expenditure_j, 0.8)
        self.assertAlmostEqual(
            env.get_queue(QueueStage.COMPUTE_UAV, 0).native_backlog, 2.0
        )

    def test_backhaul_energy_is_charged_to_uav_once(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(10.0,),
            uav_backhaul_bps=10.0,
            cloud_cpu_hz=0.0,
            uav_batteries=(BatteryConfig(20.0, 20.0),),
            uav_energy=electronics_only(backhaul_tx_power_w=5.0),
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(
            0,
            input_bits=10.0,
            cpu_cycles=10.0,
            deadline=10.0,
            partitionable=True,
        )
        env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(
                        0, task_id, Destination.uav(0), uav_fraction=0.0
                    ),
                )
            )
        )

        result = env.step()

        self.assertAlmostEqual(result.uav_energy_j[0], 5.0)
        self.assertAlmostEqual(result.battery_transitions[0].expenditure_j, 5.0)
        self.assertAlmostEqual(result.energy.uav_j[0], 5.0)
        self.assertAlmostEqual(result.energy.cloud_reception_j, 2.0)
        self.assertAlmostEqual(result.energy.total_j, 7.0)
        self.assertAlmostEqual(
            env.get_queue(QueueStage.BACKHAUL_UAV, 0).native_backlog, 0.0
        )

    def test_unaffordable_contingency_is_explicit_and_masks_dispatch(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0,),
            uav_batteries=(BatteryConfig(2.0, 1.0),),
            uav_energy=electronics_only(control_power_w=2.0),
        )
        env = VecQueueEnv(config)
        result = env.step()
        self.assertTrue(result.physical_feasibility_failure)
        self.assertFalse(result.motion_repair.feasible)
        self.assertEqual(result.battery_transitions[0].charge_after_j, 1.0)

        env.reset()
        task_id = env.submit_task(0, input_bits=1.0, cpu_cycles=1.0, deadline=10.0)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            env.step(
                StepAction(
                    dispatches=(
                        DispatchDecision(0, task_id, Destination.uav(0)),
                    )
                )
            )

    def test_reserve_shortfall_masks_uav_without_declaring_safe_hold_failure(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0,),
            uav_batteries=(BatteryConfig(5.0, 3.0),),
            uav_reserve_j=(2.0,),
            uav_energy=electronics_only(control_power_w=2.0),
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(0, input_bits=1.0, cpu_cycles=1.0, deadline=10.0)

        with self.assertRaisesRegex(ValueError, "unavailable"):
            env.step(
                StepAction(
                    dispatches=(
                        DispatchDecision(0, task_id, Destination.uav(0)),
                    )
                )
            )

        result = env.step()
        self.assertFalse(result.physical_feasibility_failure)
        self.assertEqual(result.executed_uav_cpu_frequencies_hz, (0.0,))
        self.assertAlmostEqual(result.battery_transitions[0].expenditure_j, 2.0)

    def test_dispatch_is_held_when_motion_repair_forces_same_slot_return(self):
        limits = FlightLimits(
            x_min_m=-100.0,
            x_max_m=100.0,
            y_min_m=-100.0,
            y_max_m=100.0,
            altitude_min_m=10.0,
            altitude_max_m=30.0,
            max_speed_mps=10.0,
            max_climb_mps=5.0,
            max_descent_mps=5.0,
            max_acceleration_mps2=100.0,
            minimum_separation_m=5.0,
        )
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0,),
            uav_positions=((0.0, 0.0, 20.0),),
            uav_landing_positions=((0.0, 0.0, 10.0),),
            uav_batteries=(BatteryConfig(30.0, 5.0),),
            uav_energy=electronics_only(
                control_power_w=1.0,
                parasitic_coefficient=0.01,
            ),
            flight_limits=limits,
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(0, input_bits=1.0, cpu_cycles=1.0, deadline=10.0)
        self.assertTrue(env.destination_masks()[0].uav[0])

        result = env.step(
            StepAction(
                dispatches=(
                    DispatchDecision(0, task_id, Destination.uav(0)),
                ),
                uav_motion_commands=(MotionCommand((10.0, 0.0, 0.0)),),
            )
        )

        self.assertFalse(result.physical_feasibility_failure)
        self.assertEqual(result.uav_modes_end, (UAVMode.RETURNING,))
        self.assertEqual(result.action_trace.dispatches, ())
        self.assertEqual(
            env.get_queue(QueueStage.INGRESS, 0).task_ids(),
            (task_id,),
        )


class EnvironmentSafetyTests(unittest.TestCase):
    @staticmethod
    def landing_limits():
        return FlightLimits(
            x_min_m=-100.0,
            x_max_m=100.0,
            y_min_m=-100.0,
            y_max_m=100.0,
            altitude_min_m=10.0,
            altitude_max_m=100.0,
            max_speed_mps=10.0,
            max_climb_mps=5.0,
            max_descent_mps=5.0,
            max_acceleration_mps2=100.0,
            minimum_separation_m=5.0,
        )

    def test_verified_return_is_executed_until_landed(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0,),
            uav_positions=((0.0, 0.0, 20.0),),
            uav_landing_positions=((0.0, 0.0, 10.0),),
            uav_batteries=(BatteryConfig(10.0, 3.5),),
            uav_energy=electronics_only(control_power_w=1.0),
            flight_limits=self.landing_limits(),
        )
        env = VecQueueEnv(config)

        first = env.step()
        self.assertFalse(first.physical_feasibility_failure)
        self.assertEqual(first.uav_modes_start, (UAVMode.ACTIVE,))
        self.assertEqual(first.uav_modes_end, (UAVMode.RETURNING,))
        self.assertEqual(first.executed_uav_cpu_frequencies_hz, (0.0,))
        self.assertAlmostEqual(first.uav_positions_end[0][2], 15.0)
        self.assertAlmostEqual(first.uav_energy_j[0], 1.0)
        self.assertTrue(first.return_plans[0].feasible)
        self.assertAlmostEqual(first.return_reserve_j[0], 2.0)

        task_id = env.submit_task(0, input_bits=1.0, cpu_cycles=1.0, deadline=10.0)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            env.step(
                StepAction(
                    dispatches=(
                        DispatchDecision(0, task_id, Destination.uav(0)),
                    )
                )
            )

        second = env.step()
        third = env.step()
        self.assertEqual(second.uav_modes_end, (UAVMode.RETURNING,))
        self.assertEqual(third.uav_modes_end, (UAVMode.LANDED,))
        self.assertAlmostEqual(third.uav_positions_end[0][2], 10.0)

        charge_at_landing = env.batteries[0].charge_j
        landed = env.step()
        self.assertEqual(landed.uav_modes_end, (UAVMode.LANDED,))
        self.assertAlmostEqual(landed.uav_energy_j[0], 0.0)
        self.assertAlmostEqual(env.batteries[0].charge_j, charge_at_landing)

    def test_underfunded_verified_return_is_physical_failure(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0,),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0,),
            uav_positions=((0.0, 0.0, 20.0),),
            uav_landing_positions=((0.0, 0.0, 10.0),),
            uav_batteries=(BatteryConfig(10.0, 2.5),),
            uav_energy=electronics_only(control_power_w=1.0),
            flight_limits=self.landing_limits(),
        )

        result = VecQueueEnv(config).step()

        self.assertTrue(result.physical_feasibility_failure)
        self.assertEqual(result.uav_modes_end, (UAVMode.FAILED,))
        self.assertFalse(result.motion_repair.feasible)

    def test_crossing_future_return_paths_are_rejected(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0, 1.0),
            rsu_uplink_bps=(),
            uav_uplink_bps=(1.0, 1.0),
            uav_positions=((-10.0, 0.0, 20.0), (0.0, -10.0, 20.0)),
            uav_landing_positions=((10.0, 0.0, 10.0), (0.0, 10.0, 10.0)),
            uav_batteries=(
                BatteryConfig(100.0, 100.0),
                BatteryConfig(100.0, 100.0),
            ),
            uav_energy=electronics_only(control_power_w=1.0),
            flight_limits=self.landing_limits(),
        )

        result = VecQueueEnv(config).step()

        self.assertTrue(result.physical_feasibility_failure)
        self.assertEqual(result.uav_modes_end, (UAVMode.FAILED, UAVMode.FAILED))

    def test_environment_logs_joint_mid_slot_collision_repair(self):
        limits = FlightLimits(
            x_min_m=-100.0,
            x_max_m=100.0,
            y_min_m=-100.0,
            y_max_m=100.0,
            altitude_min_m=10.0,
            altitude_max_m=100.0,
            max_speed_mps=30.0,
            max_climb_mps=10.0,
            max_descent_mps=10.0,
            max_acceleration_mps2=100.0,
            minimum_separation_m=5.0,
        )
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(),
            uav_cpu_hz=(1.0, 1.0),
            rsu_uplink_bps=(),
            uav_uplink_bps=(0.0, 0.0),
            uav_positions=((-10.0, 0.0, 20.0), (0.0, -10.0, 20.0)),
            flight_limits=limits,
        )
        env = VecQueueEnv(config)
        commands = (
            MotionCommand((20.0, 0.0, 0.0)),
            MotionCommand((0.0, 20.0, 0.0)),
        )

        result = env.step(StepAction(uav_motion_commands=commands))

        self.assertTrue(result.motion_repair.feasible)
        self.assertTrue(any(result.motion_repair.repaired))
        self.assertTrue(
            any(
                magnitude > 0
                for magnitude in result.motion_repair.correction_magnitudes_mps
            )
        )
        self.assertGreaterEqual(result.motion_repair.minimum_separation_m, 5.0)
        self.assertEqual(
            result.motion_repair.proposed_velocities,
            tuple(command.velocity for command in commands),
        )
        self.assertEqual(result.uav_positions_end, env.uav_positions)


if __name__ == "__main__":
    unittest.main()
