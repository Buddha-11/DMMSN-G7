"""Deterministic, energy-costed UAV return and landing plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .energy import SlotEnergyDemand, UAVEnergyModel
from .mobility import Position3D
from .safety import (
    FlightSafetyController,
    MotionCommand,
    UAVKinematicState,
    Velocity3D,
)


def _norm(vector: Velocity3D) -> float:
    return sqrt(sum(value * value for value in vector))


def _delta(first: Position3D, second: Position3D) -> Velocity3D:
    return tuple(a - b for a, b in zip(first, second))  # type: ignore[return-value]


@dataclass(frozen=True)
class ReturnPlan:
    """A sequence whose every segment passed the configured flight checks."""

    feasible: bool
    target_position: Position3D
    commands: tuple[MotionCommand, ...]
    executed_velocities: tuple[Velocity3D, ...]
    end_positions: tuple[Position3D, ...]
    energy_j: float
    reason: str | None = None

    @property
    def slots(self) -> int:
        return len(self.executed_velocities)

    @property
    def first_command(self) -> MotionCommand | None:
        return self.commands[0] if self.commands else None


class ReturnLandingPlanner:
    """Construct a bounded feedback trajectory to a declared landing site.

    The planner advances the same acceleration, speed, altitude, and map
    projection used by the environment. Energy is summed over the executed
    segments, so a plan is both kinematically checked and directly reservable.
    """

    def __init__(
        self,
        safety: FlightSafetyController,
        energy: UAVEnergyModel,
        slot_duration: float,
        *,
        position_tolerance_m: float = 0.1,
        velocity_tolerance_mps: float = 0.1,
        max_slots: int = 10_000,
    ) -> None:
        if position_tolerance_m <= 0 or velocity_tolerance_mps < 0:
            raise ValueError("landing tolerances are invalid")
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self.safety = safety
        self.energy = energy
        self.slot_duration = slot_duration
        self.position_tolerance_m = position_tolerance_m
        self.velocity_tolerance_mps = velocity_tolerance_mps
        self.max_slots = max_slots

    def plan(self, state: UAVKinematicState, target: Position3D) -> ReturnPlan:
        if not self._valid_target(target):
            return ReturnPlan(False, target, (), (), (), 0.0, "invalid landing site")

        current = state
        commands: list[MotionCommand] = []
        velocities: list[Velocity3D] = []
        positions: list[Position3D] = []
        total_energy = 0.0
        for _ in range(self.max_slots):
            if self.is_landed(current, target):
                return ReturnPlan(
                    True,
                    target,
                    tuple(commands),
                    tuple(velocities),
                    tuple(positions),
                    total_energy,
                )
            command = MotionCommand(self._desired_velocity(current, target))
            repaired = self.safety.repair(
                (current,),
                (command,),
                self.slot_duration,
                allow_min_altitude_arrival=(True,),
            )
            if not repaired.feasible:
                return ReturnPlan(
                    False,
                    target,
                    tuple(commands),
                    tuple(velocities),
                    tuple(positions),
                    total_energy,
                    "no feasible return segment",
                )
            velocity = repaired.executed_velocities[0]
            end = repaired.end_positions[0]
            if end == current.position and velocity == current.velocity:
                return ReturnPlan(
                    False,
                    target,
                    tuple(commands),
                    tuple(velocities),
                    tuple(positions),
                    total_energy,
                    "return planner made no progress",
                )
            total_energy += self.energy.slot_energy(
                SlotEnergyDemand(
                    horizontal_speed_mps=sqrt(velocity[0] ** 2 + velocity[1] ** 2),
                    vertical_speed_mps=velocity[2],
                    control_active_seconds=self.slot_duration,
                    airborne=True,
                ),
                self.slot_duration,
            ).total_j
            commands.append(command)
            velocities.append(velocity)
            positions.append(end)
            current = UAVKinematicState(end, velocity)

        return ReturnPlan(
            False,
            target,
            tuple(commands),
            tuple(velocities),
            tuple(positions),
            total_energy,
            "return plan exceeded max_slots",
        )

    def is_landed(self, state: UAVKinematicState, target: Position3D) -> bool:
        return (
            _norm(_delta(state.position, target)) <= self.position_tolerance_m
            and _norm(state.velocity) <= self.velocity_tolerance_mps
        )

    def _desired_velocity(
        self, state: UAVKinematicState, target: Position3D
    ) -> Velocity3D:
        difference = _delta(target, state.position)
        distance = _norm(difference)
        if distance <= self.position_tolerance_m:
            return (0.0, 0.0, 0.0)

        limits = self.safety.limits
        acceleration = limits.max_acceleration_mps2
        if acceleration > 0:
            braking_speed = sqrt(2.0 * acceleration * distance)
        else:
            braking_speed = 0.0
        requested_speed = min(
            limits.max_speed_mps,
            braking_speed,
            distance / self.slot_duration,
        )
        if requested_speed <= 0:
            return (0.0, 0.0, 0.0)
        velocity = tuple(
            requested_speed * component / distance for component in difference
        )
        horizontal = sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        vertical = velocity[2]
        vertical_limit = (
            limits.max_climb_mps if vertical >= 0 else limits.max_descent_mps
        )
        scale = 1.0
        if abs(vertical) > vertical_limit and abs(vertical) > 0:
            scale = min(scale, vertical_limit / abs(vertical))
        if horizontal**2 + vertical**2 > limits.max_speed_mps**2:
            scale = min(
                scale,
                limits.max_speed_mps
                / sqrt(horizontal**2 + vertical**2),
            )
        return tuple(scale * value for value in velocity)  # type: ignore[return-value]

    def _valid_target(self, target: Position3D) -> bool:
        limits = self.safety.limits
        return (
            limits.x_min_m <= target[0] <= limits.x_max_m
            and limits.y_min_m <= target[1] <= limits.y_max_m
            and abs(target[2] - limits.altitude_min_m) <= 1e-9
        )
