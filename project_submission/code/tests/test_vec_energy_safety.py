"""Analytical tests for Phase-2 energy and flight-safety primitives."""

import unittest

from src.vec.energy import (
    BatteryConfig,
    BatteryState,
    SlotEnergyDemand,
    UAVEnergyModel,
)
from src.vec.safety import (
    FlightLimits,
    FlightSafetyController,
    MotionCommand,
    UAVKinematicState,
)


class EnergyTests(unittest.TestCase):
    def test_landed_uav_has_no_propulsion_cost(self):
        model = UAVEnergyModel()
        airborne = model.slot_energy(SlotEnergyDemand(airborne=True), 1.0)
        landed = model.slot_energy(SlotEnergyDemand(airborne=False), 1.0)
        self.assertGreater(airborne.propulsion_j, 0.0)
        self.assertEqual(landed.propulsion_j, 0.0)

    def test_computation_energy_is_counted_once(self):
        model = UAVEnergyModel()
        result = model.slot_energy(
            SlotEnergyDemand(cpu_frequency_hz=2.0e9, computed_cycles=1.0e9),
            1.0,
        )
        self.assertAlmostEqual(result.computation_j, 0.4)
        self.assertAlmostEqual(
            result.total_j,
            result.propulsion_j
            + result.computation_j
            + result.reception_j
            + result.backhaul_j
            + result.control_j,
        )

    def test_harvest_arrives_after_expenditure_and_overflow_is_logged(self):
        battery = BatteryState(BatteryConfig(capacity_j=10.0, initial_charge_j=6.0))
        with self.assertRaisesRegex(ValueError, "cannot use end-of-slot harvest"):
            battery.apply(expenditure_j=7.0, harvest_j=100.0)

        transition = battery.apply(expenditure_j=2.0, harvest_j=9.0)
        self.assertEqual(transition.charge_after_j, 10.0)
        self.assertEqual(transition.overflow_j, 3.0)

    def test_normal_service_respects_return_reserve(self):
        battery = BatteryState(BatteryConfig(capacity_j=100.0, initial_charge_j=60.0))
        self.assertTrue(battery.can_execute(20.0, reserve_j=40.0))
        self.assertFalse(battery.can_execute(21.0, reserve_j=40.0))


class SafetyTests(unittest.TestCase):
    @staticmethod
    def limits() -> FlightLimits:
        return FlightLimits(
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

    def test_closest_approach_detects_mid_slot_crossing(self):
        distance = FlightSafetyController.closest_approach_distance(
            (-10.0, 0.0, 20.0),
            (20.0, 0.0, 0.0),
            (0.0, -10.0, 20.0),
            (0.0, 20.0, 0.0),
            1.0,
        )
        self.assertAlmostEqual(distance, 0.0)

    def test_joint_repair_avoids_collision(self):
        controller = FlightSafetyController(self.limits())
        states = (
            UAVKinematicState((-10.0, 0.0, 20.0)),
            UAVKinematicState((0.0, -10.0, 20.0)),
        )
        commands = (
            MotionCommand((20.0, 0.0, 0.0)),
            MotionCommand((0.0, 20.0, 0.0)),
        )
        result = controller.repair(states, commands, 1.0)

        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.minimum_separation_m, 5.0)
        self.assertTrue(any(result.repaired))

    def test_projection_enforces_speed_altitude_and_map_bounds(self):
        controller = FlightSafetyController(self.limits())
        state = UAVKinematicState((95.0, 95.0, 95.0))
        result = controller.repair(
            (state,), (MotionCommand((100.0, 100.0, 100.0)),), 1.0
        )
        end = result.end_positions[0]
        speed = sum(value * value for value in result.executed_velocities[0]) ** 0.5
        self.assertLessEqual(speed, 30.0)
        self.assertLessEqual(end[0], 100.0)
        self.assertLessEqual(end[1], 100.0)
        self.assertLessEqual(end[2], 100.0)

    def test_repair_preserves_enough_room_to_brake_before_map_boundary(self):
        limits = FlightLimits(
            x_min_m=0.0,
            x_max_m=100.0,
            y_min_m=0.0,
            y_max_m=100.0,
            altitude_min_m=10.0,
            altitude_max_m=100.0,
            max_speed_mps=20.0,
            max_climb_mps=5.0,
            max_descent_mps=5.0,
            max_acceleration_mps2=10.0,
            minimum_separation_m=0.0,
        )
        controller = FlightSafetyController(limits)
        state = UAVKinematicState((80.0, 50.0, 50.0))
        command = MotionCommand((20.0, 0.0, 0.0))

        for _ in range(8):
            result = controller.repair((state,), (command,), 1.0)
            self.assertTrue(result.feasible)
            self.assertLessEqual(result.end_positions[0][0], 100.0)
            state = UAVKinematicState(
                result.end_positions[0], result.executed_velocities[0]
            )


if __name__ == "__main__":
    unittest.main()
