"""Normalized Lyapunov drift-plus-penalty accounting for VEC control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LyapunovConfig:
    queue_reference_cycles: float = 1.0e9
    energy_queue_reference_j: float = 1.0e3
    energy_cost_reference_j: float = 1.0e3
    latency_reference_s: float = 1.0
    tradeoff_v: float = 1.0
    latency_weight: float = 1.0
    energy_weight: float = 1.0
    deadline_weight: float = 1.0
    physical_queue_weight: float = 1.0
    virtual_energy_weight: float = 1.0
    net_energy_target_j: tuple[float, ...] = ()
    expected_harvest_j: tuple[float, ...] = ()
    flow_tolerance_cycles: float = 1.0e-5
    flow_relative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        positive = (
            self.queue_reference_cycles,
            self.energy_queue_reference_j,
            self.energy_cost_reference_j,
            self.latency_reference_s,
        )
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Lyapunov reference scales must be finite and positive")
        nonnegative = (
            self.tradeoff_v,
            self.latency_weight,
            self.energy_weight,
            self.deadline_weight,
            self.physical_queue_weight,
            self.virtual_energy_weight,
            self.flow_tolerance_cycles,
            self.flow_relative_tolerance,
            *self.net_energy_target_j,
            *self.expected_harvest_j,
        )
        if any(not isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("Lyapunov weights, targets, and harvest must be non-negative")


@dataclass(frozen=True)
class LyapunovSlotInput:
    queue_before: Mapping[str, float]
    queue_after: Mapping[str, float]
    queue_arrivals: Mapping[str, float]
    queue_departures: Mapping[str, float]
    uav_energy_j: tuple[float, ...]
    harvest_j: tuple[float, ...]
    system_energy_j: float
    unfinished_tasks: int
    new_deadline_misses: int
    slot_duration_s: float
    external_arrival_cycles: float
    compute_service_cycles: float


@dataclass(frozen=True)
class LyapunovScore:
    immediate_cost: float
    latency_cost: float
    energy_cost: float
    deadline_cost: float
    physical_queue_term: float
    virtual_energy_term: float
    drift_plus_penalty: float
    reward: float
    exact_lyapunov_drift: float
    virtual_energy_before_j: tuple[float, ...]
    virtual_energy_after_j: tuple[float, ...]
    net_energy_increment_j: tuple[float, ...]
    normalized_queue_before: Mapping[str, float]
    normalized_queue_after: Mapping[str, float]
    flow_balance_residual_cycles: float
    used_expected_harvest: bool


class LyapunovTracker:
    """Stateful virtual-energy queues with side-effect-free candidate scoring."""

    def __init__(self, uav_count: int, config: LyapunovConfig | None = None) -> None:
        if uav_count < 0:
            raise ValueError("uav_count must be non-negative")
        self.config = config or LyapunovConfig()
        self.uav_count = uav_count
        self._targets = self._expand(
            self.config.net_energy_target_j, "net_energy_target_j"
        )
        self._expected_harvest = self._expand(
            self.config.expected_harvest_j, "expected_harvest_j"
        )
        self.virtual_energy_j = [0.0 for _ in range(uav_count)]

    def reset(self) -> None:
        self.virtual_energy_j = [0.0 for _ in range(self.uav_count)]

    def record(self, slot: LyapunovSlotInput) -> LyapunovScore:
        """Score a realized transition and update with realized harvest only."""

        return self._score(slot, slot.harvest_j, update=True, expected=False)

    def preview(
        self,
        slot: LyapunovSlotInput,
        *,
        expected_harvest_j: Sequence[float] | None = None,
    ) -> LyapunovScore:
        """Score a candidate without mutation, using expected harvest."""

        harvest = (
            tuple(float(value) for value in expected_harvest_j)
            if expected_harvest_j is not None
            else self._expected_harvest
        )
        return self._score(slot, harvest, update=False, expected=True)

    def normalized_virtual_energy(self) -> tuple[float, ...]:
        reference = self.config.energy_queue_reference_j
        return tuple(value / reference for value in self.virtual_energy_j)

    def _score(
        self,
        slot: LyapunovSlotInput,
        harvest_j: Sequence[float],
        *,
        update: bool,
        expected: bool,
    ) -> LyapunovScore:
        if len(slot.uav_energy_j) != self.uav_count:
            raise ValueError("uav_energy_j must match UAV count")
        if len(harvest_j) != self.uav_count:
            raise ValueError("harvest_j must match UAV count")
        if slot.unfinished_tasks < 0 or slot.new_deadline_misses < 0:
            raise ValueError("task counts must be non-negative")
        if slot.slot_duration_s <= 0 or slot.system_energy_j < 0:
            raise ValueError("slot duration and system energy are invalid")

        cfg = self.config
        labels = sorted(set(slot.queue_before) | set(slot.queue_after))
        normalized_before = {
            label: slot.queue_before.get(label, 0.0) / cfg.queue_reference_cycles
            for label in labels
        }
        normalized_after = {
            label: slot.queue_after.get(label, 0.0) / cfg.queue_reference_cycles
            for label in labels
        }
        physical_term = sum(
            normalized_before.get(label, 0.0)
            * (
                slot.queue_arrivals.get(label, 0.0)
                - slot.queue_departures.get(label, 0.0)
            )
            / cfg.queue_reference_cycles
            for label in labels
        )

        before_virtual = tuple(self.virtual_energy_j)
        increments = tuple(
            energy - float(harvest) - target
            for energy, harvest, target in zip(
                slot.uav_energy_j, harvest_j, self._targets
            )
        )
        virtual_term = sum(
            (backlog / cfg.energy_queue_reference_j)
            * (increment / cfg.energy_queue_reference_j)
            for backlog, increment in zip(before_virtual, increments)
        )
        after_virtual = tuple(
            max(0.0, backlog + increment)
            for backlog, increment in zip(before_virtual, increments)
        )

        latency_cost = (
            cfg.latency_weight
            * slot.slot_duration_s
            * slot.unfinished_tasks
            / cfg.latency_reference_s
        )
        energy_cost = (
            cfg.energy_weight
            * slot.system_energy_j
            / cfg.energy_cost_reference_j
        )
        deadline_cost = cfg.deadline_weight * slot.new_deadline_misses
        immediate = latency_cost + energy_cost + deadline_cost
        score = (
            cfg.tradeoff_v * immediate
            + cfg.physical_queue_weight * physical_term
            + cfg.virtual_energy_weight * virtual_term
        )

        before_lyapunov = 0.5 * (
            sum(value * value for value in normalized_before.values())
            + sum(
                (value / cfg.energy_queue_reference_j) ** 2
                for value in before_virtual
            )
        )
        after_lyapunov = 0.5 * (
            sum(value * value for value in normalized_after.values())
            + sum(
                (value / cfg.energy_queue_reference_j) ** 2
                for value in after_virtual
            )
        )
        internal_arrivals = (
            sum(slot.queue_arrivals.values()) - slot.external_arrival_cycles
        )
        internal_departures = (
            sum(slot.queue_departures.values()) - slot.compute_service_cycles
        )
        flow_residual = internal_arrivals - internal_departures
        # Internal flow is obtained by subtracting gross external arrivals and
        # final compute service from queue-ledger totals.  Those totals are
        # commonly billions of cycles, so comparing only the two small
        # remainders makes ``isclose`` vulnerable to cancellation error.  Scale
        # the relative allowance by the gross ledger values while retaining the
        # configured absolute floor.  A real one-cycle imbalance still fails;
        # sub-cycle floating-point residue from large balanced flows does not.
        flow_scale = max(
            1.0,
            sum(abs(value) for value in slot.queue_arrivals.values()),
            sum(abs(value) for value in slot.queue_departures.values()),
            abs(slot.external_arrival_cycles),
            abs(slot.compute_service_cycles),
        )
        flow_tolerance = (
            cfg.flow_tolerance_cycles
            + cfg.flow_relative_tolerance * flow_scale
        )
        if abs(flow_residual) > flow_tolerance:
            raise AssertionError(
                "internal queue transfer ledger is unbalanced: "
                f"arrivals={internal_arrivals}, departures={internal_departures}, "
                f"residual={flow_residual}, tolerance={flow_tolerance}"
            )
        if update:
            self.virtual_energy_j[:] = after_virtual

        return LyapunovScore(
            immediate_cost=immediate,
            latency_cost=latency_cost,
            energy_cost=energy_cost,
            deadline_cost=deadline_cost,
            physical_queue_term=physical_term,
            virtual_energy_term=virtual_term,
            drift_plus_penalty=score,
            reward=-score,
            exact_lyapunov_drift=after_lyapunov - before_lyapunov,
            virtual_energy_before_j=before_virtual,
            virtual_energy_after_j=after_virtual,
            net_energy_increment_j=increments,
            normalized_queue_before=normalized_before,
            normalized_queue_after=normalized_after,
            flow_balance_residual_cycles=flow_residual,
            used_expected_harvest=expected,
        )

    def _expand(self, values: Sequence[float], name: str) -> tuple[float, ...]:
        if not values:
            return tuple(0.0 for _ in range(self.uav_count))
        if len(values) != self.uav_count:
            raise ValueError(f"{name} must match UAV count")
        return tuple(float(value) for value in values)
