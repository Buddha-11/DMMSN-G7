"""Three-dimensional UAV motion projection and segment safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest
from itertools import product
from math import isfinite, sqrt
from typing import Sequence

from .mobility import Position3D


Velocity3D = tuple[float, float, float]


def _norm(vector: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def _sub(first: Sequence[float], second: Sequence[float]) -> Velocity3D:
    return tuple(float(a - b) for a, b in zip(first, second))  # type: ignore[return-value]


def _add_scaled(
    position: Sequence[float], velocity: Sequence[float], duration: float
) -> Position3D:
    return tuple(
        float(coordinate + duration * speed)
        for coordinate, speed in zip(position, velocity)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class FlightLimits:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    altitude_min_m: float
    altitude_max_m: float
    max_speed_mps: float
    max_climb_mps: float
    max_descent_mps: float
    max_acceleration_mps2: float
    minimum_separation_m: float
    max_joint_candidates: int = 10_000

    def __post_init__(self) -> None:
        values = vars(self)
        if any(
            not isfinite(value)
            for name, value in values.items()
            if name != "max_joint_candidates"
        ):
            raise ValueError("flight limits must be finite")
        if not self.x_min_m < self.x_max_m or not self.y_min_m < self.y_max_m:
            raise ValueError("map bounds must have positive extent")
        if not self.altitude_min_m <= self.altitude_max_m:
            raise ValueError("altitude bounds are invalid")
        if min(
            self.max_speed_mps,
            self.max_climb_mps,
            self.max_descent_mps,
            self.max_acceleration_mps2,
            self.minimum_separation_m,
        ) < 0:
            raise ValueError("speed, acceleration, and separation limits are non-negative")
        if self.max_joint_candidates <= 0:
            raise ValueError("max_joint_candidates must be positive")


@dataclass(frozen=True)
class UAVKinematicState:
    position: Position3D
    velocity: Velocity3D = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MotionCommand:
    velocity: Velocity3D


@dataclass(frozen=True)
class MotionRepairResult:
    proposed_velocities: tuple[Velocity3D, ...]
    executed_velocities: tuple[Velocity3D, ...]
    end_positions: tuple[Position3D, ...]
    repaired: tuple[bool, ...]
    minimum_separation_m: float
    candidates_checked: int
    feasible: bool

    @property
    def correction_magnitudes_mps(self) -> tuple[float, ...]:
        return tuple(
            _norm(_sub(executed, proposed))
            for proposed, executed in zip(
                self.proposed_velocities, self.executed_velocities
            )
        )


class FlightSafetyController:
    """Finite joint search over projected velocity candidates."""

    _SCALES = (1.0, 0.75, 0.5, 0.25, 0.0)

    def __init__(self, limits: FlightLimits) -> None:
        self.limits = limits

    def repair(
        self,
        states: Sequence[UAVKinematicState],
        commands: Sequence[MotionCommand],
        slot_duration: float,
        *,
        allow_min_altitude_arrival: Sequence[bool] | None = None,
    ) -> MotionRepairResult:
        if len(states) != len(commands):
            raise ValueError("states and commands must have equal length")
        if not isfinite(slot_duration) or slot_duration <= 0:
            raise ValueError("slot_duration must be finite and positive")
        landing_flags = tuple(allow_min_altitude_arrival or (False,) * len(states))
        if len(landing_flags) != len(states):
            raise ValueError("landing flags must match UAV count")
        proposed = tuple(command.velocity for command in commands)
        candidate_sets: list[tuple[Velocity3D, ...]] = []
        for state, command, landing_allowed in zip(
            states, commands, landing_flags
        ):
            candidates: list[Velocity3D] = []
            for scale in self._SCALES:
                scaled = tuple(scale * value for value in command.velocity)
                projected = self._project_individual(state, scaled, slot_duration)
                if (
                    projected not in candidates
                    and self._is_individually_feasible(
                        state,
                        projected,
                        slot_duration,
                        allow_min_altitude_arrival=landing_allowed,
                    )
                ):
                    candidates.append(projected)
            if not candidates:
                projected = self._project_individual(
                    state, command.velocity, slot_duration
                )
                failed_index = len(candidate_sets)
                failure_velocities = tuple(
                    projected if index == failed_index else (0.0, 0.0, 0.0)
                    for index in range(len(states))
                )
                ends = tuple(
                    _add_scaled(item.position, velocity, slot_duration)
                    for item, velocity in zip(states, failure_velocities)
                )
                return MotionRepairResult(
                    proposed_velocities=proposed,
                    executed_velocities=failure_velocities,
                    end_positions=ends,
                    repaired=tuple(True for _ in proposed),
                    minimum_separation_m=0.0,
                    candidates_checked=0,
                    feasible=False,
                )
            candidate_sets.append(tuple(candidates))

        def deviation(combination: Sequence[Velocity3D]) -> float:
            return sum(
                _norm(_sub(executed, requested)) ** 2
                for executed, requested in zip(combination, proposed)
            )

        ranked = nsmallest(
            self.limits.max_joint_candidates,
            product(*candidate_sets),
            key=deviation,
        )
        checked = 0
        for combination in ranked:
            checked += 1
            minimum = self.minimum_pair_separation(
                states, combination, slot_duration
            )
            if minimum + 1e-9 < self.limits.minimum_separation_m:
                continue
            ends = tuple(
                _add_scaled(state.position, velocity, slot_duration)
                for state, velocity in zip(states, combination)
            )
            return MotionRepairResult(
                proposed_velocities=proposed,
                executed_velocities=tuple(combination),
                end_positions=ends,
                repaired=tuple(
                    _norm(_sub(executed, requested)) > 1e-9
                    for executed, requested in zip(combination, proposed)
                ),
                minimum_separation_m=minimum,
                candidates_checked=checked,
                feasible=True,
            )

        individually_projected = tuple(candidates[0] for candidates in candidate_sets)
        ends = tuple(
            _add_scaled(state.position, velocity, slot_duration)
            for state, velocity in zip(states, individually_projected)
        )
        return MotionRepairResult(
            proposed_velocities=proposed,
            executed_velocities=individually_projected,
            end_positions=ends,
            repaired=tuple(True for _ in proposed),
            minimum_separation_m=self.minimum_pair_separation(
                states, individually_projected, slot_duration
            ),
            candidates_checked=checked,
            feasible=False,
        )

    def _is_individually_feasible(
        self,
        state: UAVKinematicState,
        velocity: Velocity3D,
        duration: float,
        *,
        allow_min_altitude_arrival: bool = False,
    ) -> bool:
        limits = self.limits
        if _norm(velocity) > limits.max_speed_mps + 1e-9:
            return False
        if not -limits.max_descent_mps - 1e-9 <= velocity[2] <= limits.max_climb_mps + 1e-9:
            return False
        acceleration = _norm(_sub(velocity, state.velocity)) / duration
        if acceleration > limits.max_acceleration_mps2 + 1e-9:
            return False
        end = _add_scaled(state.position, velocity, duration)
        inside = (
            limits.x_min_m - 1e-9 <= end[0] <= limits.x_max_m + 1e-9
            and limits.y_min_m - 1e-9 <= end[1] <= limits.y_max_m + 1e-9
            and limits.altitude_min_m - 1e-9
            <= end[2]
            <= limits.altitude_max_m + 1e-9
        )
        return inside and self._has_braking_reserve(
            end,
            velocity,
            allow_min_altitude_arrival=allow_min_altitude_arrival,
        )

    def _has_braking_reserve(
        self,
        position: Position3D,
        velocity: Velocity3D,
        *,
        allow_min_altitude_arrival: bool,
    ) -> bool:
        """Keep the next state inside a conservative braking-safe set.

        Merely clamping the next endpoint can leave a UAV at a map boundary
        with nonzero outward speed. The following slot then has no command that
        satisfies both the boundary and acceleration limits. Requiring the
        continuous stopping distance to fit along the current velocity ray
        prevents that one-step-later infeasibility.
        """

        speed = _norm(velocity)
        if speed <= 1.0e-12:
            return True
        acceleration = self.limits.max_acceleration_mps2
        if acceleration <= 0.0:
            return False
        direction = tuple(value / speed for value in velocity)
        lower = (
            self.limits.x_min_m,
            self.limits.y_min_m,
            self.limits.altitude_min_m,
        )
        upper = (
            self.limits.x_max_m,
            self.limits.y_max_m,
            self.limits.altitude_max_m,
        )
        ray_distances = []
        for axis, (coordinate, component, minimum, maximum) in enumerate(
            zip(position, direction, lower, upper)
        ):
            if component > 1.0e-12:
                ray_distances.append((maximum - coordinate) / component)
            elif component < -1.0e-12:
                if not (axis == 2 and allow_min_altitude_arrival):
                    ray_distances.append((coordinate - minimum) / -component)
        distance_to_boundary = min(ray_distances, default=float("inf"))
        stopping_distance = speed * speed / (2.0 * acceleration)
        return stopping_distance <= distance_to_boundary + 1.0e-9

    def _project_individual(
        self,
        state: UAVKinematicState,
        velocity: Sequence[float],
        duration: float,
    ) -> Velocity3D:
        vx, vy, vz = (float(value) for value in velocity)
        vz = min(max(vz, -self.limits.max_descent_mps), self.limits.max_climb_mps)
        speed = _norm((vx, vy, vz))
        if speed > self.limits.max_speed_mps and speed > 0:
            factor = self.limits.max_speed_mps / speed
            vx, vy, vz = vx * factor, vy * factor, vz * factor

        delta = (vx - state.velocity[0], vy - state.velocity[1], vz - state.velocity[2])
        max_delta = self.limits.max_acceleration_mps2 * duration
        delta_norm = _norm(delta)
        if delta_norm > max_delta and delta_norm > 0:
            factor = max_delta / delta_norm
            vx = state.velocity[0] + delta[0] * factor
            vy = state.velocity[1] + delta[1] * factor
            vz = state.velocity[2] + delta[2] * factor

        end = _add_scaled(state.position, (vx, vy, vz), duration)
        clamped = (
            min(max(end[0], self.limits.x_min_m), self.limits.x_max_m),
            min(max(end[1], self.limits.y_min_m), self.limits.y_max_m),
            min(
                max(end[2], self.limits.altitude_min_m),
                self.limits.altitude_max_m,
            ),
        )
        return tuple(
            (coordinate - start) / duration
            for coordinate, start in zip(clamped, state.position)
        )  # type: ignore[return-value]

    @staticmethod
    def closest_approach_distance(
        first_position: Position3D,
        first_velocity: Velocity3D,
        second_position: Position3D,
        second_velocity: Velocity3D,
        duration: float,
    ) -> float:
        relative_position = _sub(first_position, second_position)
        relative_velocity = _sub(first_velocity, second_velocity)
        speed_squared = sum(value * value for value in relative_velocity)
        if speed_squared <= 1e-18:
            closest_time = 0.0
        else:
            closest_time = -sum(
                position * velocity
                for position, velocity in zip(
                    relative_position, relative_velocity
                )
            ) / speed_squared
            closest_time = min(max(closest_time, 0.0), duration)
        separation = tuple(
            position + closest_time * velocity
            for position, velocity in zip(relative_position, relative_velocity)
        )
        return _norm(separation)

    @classmethod
    def minimum_pair_separation(
        cls,
        states: Sequence[UAVKinematicState],
        velocities: Sequence[Velocity3D],
        duration: float,
    ) -> float:
        if len(states) < 2:
            return float("inf")
        minimum = float("inf")
        for first in range(len(states)):
            for second in range(first + 1, len(states)):
                minimum = min(
                    minimum,
                    cls.closest_approach_distance(
                        states[first].position,
                        velocities[first],
                        states[second].position,
                        velocities[second],
                        duration,
                    ),
                )
        return minimum
