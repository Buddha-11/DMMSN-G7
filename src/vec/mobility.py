"""Road-constrained mobility providers for the VEC simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence

import numpy as np


Position3D = tuple[float, float, float]


class MobilityProvider(Protocol):
    """Minimal interface shared by synthetic and future SUMO mobility."""

    def reset(self) -> None: ...

    def positions(self) -> tuple[Position3D, ...]: ...

    def advance(self, duration: float) -> tuple[Position3D, ...]: ...


class RecordedTraceMobility:
    """Deterministic replay of complete SUMO/exported position frames."""

    def __init__(
        self,
        frames: Sequence[Sequence[Position3D]],
        slot_duration: float,
    ) -> None:
        if not frames:
            raise ValueError("recorded mobility requires at least one frame")
        if not isfinite(slot_duration) or slot_duration <= 0:
            raise ValueError("recorded mobility slot duration must be positive")
        vehicle_count = len(frames[0])
        if vehicle_count <= 0 or any(len(frame) != vehicle_count for frame in frames):
            raise ValueError("all recorded frames need the same positive vehicle count")
        self._frames = tuple(
            tuple(SyntheticRoadMobility._position(position) for position in frame)
            for frame in frames
        )
        self.slot_duration = float(slot_duration)
        self.vehicle_count = vehicle_count
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def positions(self) -> tuple[Position3D, ...]:
        return self._frames[self._index]

    def advance(self, duration: float) -> tuple[Position3D, ...]:
        if abs(float(duration) - self.slot_duration) > 1.0e-9:
            raise ValueError("environment slot does not match recorded trace interval")
        # Holding the final recorded position makes a finite arrival trace
        # drainable without inventing or looping new mobility observations.
        self._index = min(self._index + 1, len(self._frames) - 1)
        return self.positions()


@dataclass(frozen=True)
class PolylineRoute:
    """A vehicle route represented by connected straight road segments."""

    points: tuple[Position3D, ...]
    loop: bool = False

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("a route requires at least two points")
        points = np.asarray(self.points, dtype=float)
        if points.shape != (len(self.points), 3) or not np.isfinite(points).all():
            raise ValueError("route points must be finite 3-D coordinates")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError("consecutive route points must be distinct")

    @property
    def length(self) -> float:
        points = self._path_points()
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def position_at(self, distance: float) -> Position3D:
        """Interpolate a position at a clamped distance along the route."""

        if not isfinite(distance):
            raise ValueError("route distance must be finite")
        distance = min(max(float(distance), 0.0), self.length)
        points = self._path_points()
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        traversed = 0.0
        for index, segment_length in enumerate(lengths):
            segment_end = traversed + float(segment_length)
            if distance <= segment_end or index == len(lengths) - 1:
                fraction = (distance - traversed) / float(segment_length)
                position = points[index] + fraction * (points[index + 1] - points[index])
                return tuple(float(value) for value in position)
            traversed = segment_end
        raise AssertionError("unreachable route interpolation state")

    def _path_points(self) -> np.ndarray:
        points = np.asarray(self.points, dtype=float)
        if self.loop and not np.array_equal(points[0], points[-1]):
            points = np.vstack((points, points[0]))
        return points


@dataclass(frozen=True)
class VehicleRouteConfig:
    route: PolylineRoute
    speed_mps: float
    initial_distance_m: float = 0.0
    initial_direction: int = 1

    def __post_init__(self) -> None:
        if not isfinite(self.speed_mps) or self.speed_mps < 0:
            raise ValueError("vehicle speed must be finite and non-negative")
        if not isfinite(self.initial_distance_m):
            raise ValueError("initial route distance must be finite")
        if not 0 <= self.initial_distance_m <= self.route.length:
            raise ValueError("initial route distance lies outside the route")
        if self.initial_direction not in (-1, 1):
            raise ValueError("initial_direction must be -1 or 1")


class SyntheticRoadMobility:
    """Deterministic motion along polylines with loop or endpoint reflection."""

    def __init__(
        self,
        vehicle_count: int,
        routes: Sequence[VehicleRouteConfig | None] = (),
        static_positions: Sequence[Position3D] = (),
    ) -> None:
        if vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive")
        if routes and len(routes) != vehicle_count:
            raise ValueError("routes must have one entry per vehicle")
        if static_positions and len(static_positions) != vehicle_count:
            raise ValueError("static_positions must have one entry per vehicle")
        self.vehicle_count = vehicle_count
        self._routes = tuple(routes) if routes else (None,) * vehicle_count
        self._static_positions = (
            tuple(self._position(position) for position in static_positions)
            if static_positions
            else ((0.0, 0.0, 0.0),) * vehicle_count
        )
        self._distances: list[float] = []
        self._directions: list[int] = []
        self.reset()

    def reset(self) -> None:
        self._distances = [
            0.0 if route is None else route.initial_distance_m
            for route in self._routes
        ]
        self._directions = [
            1 if route is None else route.initial_direction for route in self._routes
        ]

    def positions(self) -> tuple[Position3D, ...]:
        result: list[Position3D] = []
        for index, route_config in enumerate(self._routes):
            if route_config is None:
                result.append(self._static_positions[index])
            else:
                result.append(route_config.route.position_at(self._distances[index]))
        return tuple(result)

    def advance(self, duration: float) -> tuple[Position3D, ...]:
        if not isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")
        for index, route_config in enumerate(self._routes):
            if route_config is None or route_config.speed_mps == 0:
                continue
            distance = (
                self._distances[index]
                + self._directions[index] * route_config.speed_mps * duration
            )
            length = route_config.route.length
            if route_config.route.loop:
                distance %= length
            else:
                # Reflect repeatedly so even a large slot displacement remains
                # on the declared road rather than jumping beyond an endpoint.
                while distance < 0 or distance > length:
                    if distance > length:
                        distance = 2 * length - distance
                        self._directions[index] *= -1
                    elif distance < 0:
                        distance = -distance
                        self._directions[index] *= -1
            self._distances[index] = distance
        return self.positions()

    @staticmethod
    def _position(position: Sequence[float]) -> Position3D:
        values = tuple(float(value) for value in position)
        if len(values) != 3 or not all(isfinite(value) for value in values):
            raise ValueError("positions must be finite 3-D coordinates")
        return values  # type: ignore[return-value]
