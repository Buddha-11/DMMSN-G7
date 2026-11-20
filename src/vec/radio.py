"""Deterministic V2R/V2U channels and queue-aware OFDMA scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, exp, isfinite, log2, pi, sqrt
from typing import Sequence

from .mobility import Position3D


@dataclass(frozen=True)
class RadioConfig:
    """Parameters shared by all controllers in geometry-aware experiments."""

    vehicle_tx_power_w: float = 0.4
    noise_psd_w_hz: float = 3.981071705534985e-21  # -174 dBm/Hz
    reference_gain: float = 1e-4
    terrestrial_path_loss_exponent: float = 2.5
    aerial_path_loss_exponent: float = 2.2
    los_a: float = 9.61
    los_b: float = 0.16
    los_extra_loss_db: float = 1.0
    nlos_extra_loss_db: float = 20.0
    minimum_distance_m: float = 1.0
    rsu_coverage_m: float = 500.0
    uav_coverage_m: float = 300.0
    age_priority_cap: float = 4.0

    def __post_init__(self) -> None:
        positive = (
            self.vehicle_tx_power_w,
            self.noise_psd_w_hz,
            self.reference_gain,
            self.terrestrial_path_loss_exponent,
            self.aerial_path_loss_exponent,
            self.los_a,
            self.los_b,
            self.minimum_distance_m,
            self.rsu_coverage_m,
            self.uav_coverage_m,
            self.age_priority_cap,
        )
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("radio parameters must be finite and positive")
        if self.los_extra_loss_db < 0 or self.nlos_extra_loss_db < 0:
            raise ValueError("extra path losses must be non-negative dB values")


@dataclass(frozen=True)
class LinkRequest:
    task_id: int
    vehicle_id: int
    vehicle_position: Position3D
    remaining_bits: float
    equivalent_cycles: float
    age_seconds: float
    deadline_seconds: float


@dataclass(frozen=True)
class LinkAllocation:
    task_id: int
    vehicle_id: int
    bandwidth_hz: float
    channel_gain: float
    rate_bps: float
    bit_budget: float
    priority: float


class DeterministicChannelModel:
    """Large-scale deterministic channel model used by the queue simulator."""

    def __init__(self, config: RadioConfig | None = None) -> None:
        self.config = config or RadioConfig()

    @staticmethod
    def distance(first: Position3D, second: Position3D) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))

    @staticmethod
    def horizontal_distance(first: Position3D, second: Position3D) -> float:
        return sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)

    def in_rsu_coverage(self, vehicle: Position3D, rsu: Position3D) -> bool:
        return self.horizontal_distance(vehicle, rsu) <= self.config.rsu_coverage_m

    def in_uav_coverage(self, vehicle: Position3D, uav: Position3D) -> bool:
        return self.horizontal_distance(vehicle, uav) <= self.config.uav_coverage_m

    def terrestrial_gain(self, vehicle: Position3D, rsu: Position3D) -> float:
        distance = max(self.distance(vehicle, rsu), self.config.minimum_distance_m)
        return self.config.reference_gain * distance ** (
            -self.config.terrestrial_path_loss_exponent
        )

    def los_probability(self, vehicle: Position3D, uav: Position3D) -> float:
        horizontal = max(
            self.horizontal_distance(vehicle, uav), self.config.minimum_distance_m
        )
        height = abs(uav[2] - vehicle[2])
        elevation_deg = atan2(height, horizontal) * 180.0 / pi
        return 1.0 / (
            1.0
            + self.config.los_a
            * exp(-self.config.los_b * (elevation_deg - self.config.los_a))
        )

    def aerial_gain(self, vehicle: Position3D, uav: Position3D) -> float:
        distance = max(self.distance(vehicle, uav), self.config.minimum_distance_m)
        base = self.config.reference_gain * distance ** (
            -self.config.aerial_path_loss_exponent
        )
        probability_los = self.los_probability(vehicle, uav)
        los_factor = 10 ** (-self.config.los_extra_loss_db / 10.0)
        nlos_factor = 10 ** (-self.config.nlos_extra_loss_db / 10.0)
        return base * (
            probability_los * los_factor
            + (1.0 - probability_los) * nlos_factor
        )

    def rate(self, gain: float, bandwidth_hz: float) -> float:
        if bandwidth_hz <= 0 or gain <= 0:
            return 0.0
        noise = self.config.noise_psd_w_hz * bandwidth_hz
        snr = self.config.vehicle_tx_power_w * gain / max(noise, 1e-30)
        return bandwidth_hz * log2(1.0 + snr)


class QueueAwareOFDMA:
    """Allocate orthogonal bandwidth by backlog and deadline age.

    Each eligible link receives a bandwidth fraction proportional to its
    equivalent-cycle backlog multiplied by a capped age/deadline factor. Ties
    are resolved by task ID through deterministic ordering.
    """

    def __init__(self, channel: DeterministicChannelModel | None = None) -> None:
        self.channel = channel or DeterministicChannelModel()

    @classmethod
    def from_config(cls, config: RadioConfig) -> "QueueAwareOFDMA":
        return cls(DeterministicChannelModel(config))

    def schedule_rsu(
        self,
        requests: Sequence[LinkRequest],
        rsu_position: Position3D,
        total_bandwidth_hz: float,
        slot_duration: float,
    ) -> tuple[LinkAllocation, ...]:
        return self._schedule(
            requests,
            rsu_position,
            total_bandwidth_hz,
            slot_duration,
            aerial=False,
        )

    def schedule_uav(
        self,
        requests: Sequence[LinkRequest],
        uav_position: Position3D,
        total_bandwidth_hz: float,
        slot_duration: float,
    ) -> tuple[LinkAllocation, ...]:
        return self._schedule(
            requests,
            uav_position,
            total_bandwidth_hz,
            slot_duration,
            aerial=True,
        )

    def _schedule(
        self,
        requests: Sequence[LinkRequest],
        node_position: Position3D,
        total_bandwidth_hz: float,
        slot_duration: float,
        *,
        aerial: bool,
    ) -> tuple[LinkAllocation, ...]:
        if total_bandwidth_hz < 0 or slot_duration <= 0:
            raise ValueError("bandwidth must be non-negative and duration positive")
        eligible: list[tuple[LinkRequest, float, float]] = []
        for request in sorted(requests, key=lambda item: item.task_id):
            covered = (
                self.channel.in_uav_coverage(request.vehicle_position, node_position)
                if aerial
                else self.channel.in_rsu_coverage(request.vehicle_position, node_position)
            )
            if not covered or request.remaining_bits <= 0:
                continue
            gain = (
                self.channel.aerial_gain(request.vehicle_position, node_position)
                if aerial
                else self.channel.terrestrial_gain(
                    request.vehicle_position, node_position
                )
            )
            relative_age = request.age_seconds / max(request.deadline_seconds, 1e-12)
            age_factor = 1.0 + min(
                max(relative_age, 0.0), self.channel.config.age_priority_cap
            )
            priority = max(request.equivalent_cycles, 1e-12) * age_factor
            eligible.append((request, gain, priority))
        if not eligible or total_bandwidth_hz == 0:
            return ()
        total_priority = sum(priority for _, _, priority in eligible)
        allocations: list[LinkAllocation] = []
        for request, gain, priority in eligible:
            bandwidth = total_bandwidth_hz * priority / total_priority
            rate = self.channel.rate(gain, bandwidth)
            allocations.append(
                LinkAllocation(
                    task_id=request.task_id,
                    vehicle_id=request.vehicle_id,
                    bandwidth_hz=bandwidth,
                    channel_gain=gain,
                    rate_bps=rate,
                    bit_budget=min(request.remaining_bits, rate * slot_duration),
                    priority=priority,
                )
            )
        return tuple(allocations)
