"""Complete per-run evidence and censor-aware metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

import numpy as np

from ..env import StepResult, UAVMode, VecQueueEnv


@dataclass(frozen=True)
class SlotTrace:
    slot: int
    phase: str
    total_backlog_cycles: float
    backlog_by_stage_cycles: Mapping[str, float]
    completed_tasks: int
    new_deadline_misses: int
    cumulative_completed_tasks: int
    cumulative_deadline_misses: int
    cumulative_reward: float
    reward: float
    energy_j: Mapping[str, float]
    battery_charge_j: tuple[float, ...]
    battery_minimum_j: tuple[float, ...]
    virtual_energy_deficit_j: tuple[float, ...]
    harvest_overflow_j: float
    uav_modes: tuple[str, ...]
    repaired_uavs: int
    minimum_separation_m: float
    physical_feasibility_failure: bool
    conservation_residual_cycles: float
    controller_seconds: float
    candidates_evaluated: int
    adaptation_updates: int = 0
    adaptation_seconds: float = 0.0


@dataclass(frozen=True)
class TaskOutcome:
    task_id: int
    vehicle_id: int
    arrival_time_s: float
    deadline_s: float
    completed: bool
    response_time_s: float | None
    completed_within_deadline: bool
    deadline_missed: bool
    censor_age_s: float | None
    destination: str | None


@dataclass(frozen=True)
class RunMetrics:
    scheduled_tasks: int
    admitted_tasks: int
    completed_tasks: int
    unfinished_censored_tasks: int
    deadline_misses: int
    deadline_success_rate: float
    completion_rate: float
    throughput_tasks_per_s: float
    response_time_mean_s: float | None
    response_time_p95_s: float | None
    response_time_p99_s: float | None
    simulated_slots: int
    offered_load_slots: int
    drain_slots_used: int
    drained_completely: bool
    realized_offered_load_ratio: float
    final_backlog_cycles: float
    peak_backlog_cycles: float
    backlog_growth_cycles_per_slot: float
    backlog_by_stage_final_cycles: Mapping[str, float]
    backlog_by_stage_peak_cycles: Mapping[str, float]
    backlog_by_stage_growth_cycles_per_slot: Mapping[str, float]
    energy_j: Mapping[str, float]
    minimum_battery_charge_j: tuple[float, ...]
    final_virtual_energy_deficit_j: tuple[float, ...]
    peak_virtual_energy_deficit_j: tuple[float, ...]
    harvest_overflow_j: float
    landing_uav_slots: int
    unavailable_uav_slots: int
    repair_rate: float
    physical_feasibility_failures: int
    battery_violations: int
    separation_violations: int
    conservation_violations: int
    cumulative_reward: float
    controller_total_seconds: float
    controller_mean_seconds: float
    controller_p95_seconds: float
    online_update_seconds: float
    online_updates: int
    offline_training_seconds: float | None
    recovery_time_slots: int | None
    adaptation_curve_reward: Mapping[str, float]


def make_slot_trace(
    result: StepResult,
    *,
    offered_load_horizon: int,
    cumulative_completed: int,
    cumulative_misses: int,
    cumulative_reward: float,
    controller_seconds: float,
    candidates_evaluated: int,
    adaptation_updates: int = 0,
    adaptation_seconds: float = 0.0,
) -> SlotTrace:
    stage_backlog: dict[str, float] = {}
    for label, value in result.queue_after.items():
        stage = label.split(":", 1)[0]
        stage_backlog[stage] = stage_backlog.get(stage, 0.0) + value
    energy = result.energy
    uav_breakdown = result.uav_energy_breakdown
    return SlotTrace(
        slot=result.slot,
        phase="offered_load" if result.slot < offered_load_horizon else "drain",
        total_backlog_cycles=sum(result.queue_after.values()),
        backlog_by_stage_cycles=stage_backlog,
        completed_tasks=len(result.completed_task_ids),
        new_deadline_misses=len(result.new_deadline_miss_ids),
        cumulative_completed_tasks=cumulative_completed,
        cumulative_deadline_misses=cumulative_misses,
        cumulative_reward=cumulative_reward,
        reward=result.lyapunov.reward,
        energy_j={
            "vehicle_compute": sum(energy.vehicle_compute_j),
            "vehicle_uplink": sum(energy.vehicle_uplink_j),
            "rsu_compute": sum(energy.rsu_compute_j),
            "rsu_reception": sum(energy.rsu_reception_j),
            "uav_total": sum(energy.uav_j),
            "uav_propulsion": sum(item.propulsion_j for item in uav_breakdown),
            "uav_computation": sum(item.computation_j for item in uav_breakdown),
            "uav_reception": sum(item.reception_j for item in uav_breakdown),
            "uav_backhaul": sum(item.backhaul_j for item in uav_breakdown),
            "uav_control": sum(item.control_j for item in uav_breakdown),
            "cloud_compute": energy.cloud_compute_j,
            "cloud_reception": energy.cloud_reception_j,
            "system_total": energy.total_j,
        },
        battery_charge_j=tuple(
            transition.charge_after_j for transition in result.battery_transitions
        ),
        battery_minimum_j=tuple(
            min(transition.charge_before_j, transition.charge_after_j)
            for transition in result.battery_transitions
        ),
        virtual_energy_deficit_j=result.lyapunov.virtual_energy_after_j,
        harvest_overflow_j=sum(
            transition.overflow_j for transition in result.battery_transitions
        ),
        uav_modes=tuple(mode.value for mode in result.uav_modes_end),
        repaired_uavs=sum(result.motion_repair.repaired),
        minimum_separation_m=result.motion_repair.minimum_separation_m,
        physical_feasibility_failure=result.physical_feasibility_failure,
        conservation_residual_cycles=result.conservation_residual,
        controller_seconds=controller_seconds,
        candidates_evaluated=candidates_evaluated,
        adaptation_updates=adaptation_updates,
        adaptation_seconds=adaptation_seconds,
    )


def summarize_run(
    environment: VecQueueEnv,
    traces: Sequence[SlotTrace],
    *,
    scheduled_tasks: int,
    offered_load_slots: int,
    realized_offered_load_ratio: float,
    offline_training_seconds: float | None = None,
    change_slot: int | None = None,
) -> tuple[RunMetrics, tuple[TaskOutcome, ...]]:
    if not traces:
        raise ValueError("at least one slot trace is required")
    end_time = traces[-1].slot * environment.config.slot_duration + environment.config.slot_duration
    outcomes: list[TaskOutcome] = []
    response_times: list[float] = []
    deadline_successes = 0
    for task_id, record in sorted(environment.tasks.items()):
        latency = record.latency
        on_time = latency is not None and latency <= record.spec.deadline
        if latency is not None:
            response_times.append(latency)
        if on_time:
            deadline_successes += 1
        destination = (
            None
            if record.destination_kind is None
            else f"{record.destination_kind.value}:{record.destination_id}"
        )
        outcomes.append(
            TaskOutcome(
                task_id=task_id,
                vehicle_id=record.spec.vehicle_id,
                arrival_time_s=record.spec.arrival_time,
                deadline_s=record.spec.deadline,
                completed=record.is_complete,
                response_time_s=latency,
                completed_within_deadline=on_time,
                deadline_missed=record.deadline_missed,
                censor_age_s=(
                    None
                    if record.is_complete
                    else max(0.0, end_time - record.spec.arrival_time)
                ),
                destination=destination,
            )
        )

    admitted = len(outcomes)
    completed = len(response_times)
    stage_names = sorted(
        {stage for trace in traces for stage in trace.backlog_by_stage_cycles}
    )
    offered = [trace for trace in traces if trace.slot < offered_load_slots]
    slope_source = offered if len(offered) >= 2 else list(traces)
    energy_names = sorted({name for trace in traces for name in trace.energy_j})
    energy = {
        name: sum(trace.energy_j.get(name, 0.0) for trace in traces)
        for name in energy_names
    }
    battery_count = max((len(trace.battery_charge_j) for trace in traces), default=0)
    virtual_count = max(
        (len(trace.virtual_energy_deficit_j) for trace in traces), default=0
    )
    minimum_battery = tuple(
        min(trace.battery_minimum_j[index] for trace in traces)
        for index in range(battery_count)
    )
    peak_virtual = tuple(
        max(trace.virtual_energy_deficit_j[index] for trace in traces)
        for index in range(virtual_count)
    )
    final_virtual = traces[-1].virtual_energy_deficit_j
    controller_times = [trace.controller_seconds for trace in traces]
    repair_denominator = len(traces) * len(environment.config.uav_cpu_hz)
    separation_limit = (
        environment.config.flight_limits.minimum_separation_m
        if environment.config.flight_limits is not None
        else None
    )
    separation_violations = (
        sum(
            trace.minimum_separation_m + 1.0e-9 < separation_limit
            for trace in traces
        )
        if separation_limit is not None
        else 0
    )
    adaptation_curve = _adaptation_curve(traces, change_slot)
    return (
        RunMetrics(
            scheduled_tasks=scheduled_tasks,
            admitted_tasks=admitted,
            completed_tasks=completed,
            unfinished_censored_tasks=admitted - completed,
            deadline_misses=sum(outcome.deadline_missed for outcome in outcomes),
            deadline_success_rate=(
                deadline_successes / admitted if admitted else 1.0
            ),
            completion_rate=completed / admitted if admitted else 1.0,
            throughput_tasks_per_s=(completed / end_time if end_time > 0 else 0.0),
            response_time_mean_s=(fmean(response_times) if response_times else None),
            response_time_p95_s=_percentile(response_times, 95.0),
            response_time_p99_s=_percentile(response_times, 99.0),
            simulated_slots=len(traces),
            offered_load_slots=offered_load_slots,
            drain_slots_used=sum(trace.phase == "drain" for trace in traces),
            drained_completely=(completed == admitted and admitted == scheduled_tasks),
            realized_offered_load_ratio=realized_offered_load_ratio,
            final_backlog_cycles=traces[-1].total_backlog_cycles,
            peak_backlog_cycles=max(trace.total_backlog_cycles for trace in traces),
            backlog_growth_cycles_per_slot=_slope(
                [trace.slot for trace in slope_source],
                [trace.total_backlog_cycles for trace in slope_source],
            ),
            backlog_by_stage_final_cycles={
                stage: traces[-1].backlog_by_stage_cycles.get(stage, 0.0)
                for stage in stage_names
            },
            backlog_by_stage_peak_cycles={
                stage: max(
                    trace.backlog_by_stage_cycles.get(stage, 0.0)
                    for trace in traces
                )
                for stage in stage_names
            },
            backlog_by_stage_growth_cycles_per_slot={
                stage: _slope(
                    [trace.slot for trace in slope_source],
                    [
                        trace.backlog_by_stage_cycles.get(stage, 0.0)
                        for trace in slope_source
                    ],
                )
                for stage in stage_names
            },
            energy_j=energy,
            minimum_battery_charge_j=minimum_battery,
            final_virtual_energy_deficit_j=final_virtual,
            peak_virtual_energy_deficit_j=peak_virtual,
            harvest_overflow_j=sum(trace.harvest_overflow_j for trace in traces),
            landing_uav_slots=sum(
                mode == UAVMode.LANDED.value
                for trace in traces
                for mode in trace.uav_modes
            ),
            unavailable_uav_slots=sum(
                mode != UAVMode.ACTIVE.value
                for trace in traces
                for mode in trace.uav_modes
            ),
            repair_rate=(
                sum(trace.repaired_uavs for trace in traces) / repair_denominator
                if repair_denominator
                else 0.0
            ),
            physical_feasibility_failures=sum(
                trace.physical_feasibility_failure for trace in traces
            ),
            battery_violations=sum(
                charge < -1.0e-9
                for trace in traces
                for charge in trace.battery_charge_j
            ),
            separation_violations=separation_violations,
            conservation_violations=sum(
                abs(trace.conservation_residual_cycles)
                > environment.config.conservation_atol
                for trace in traces
            ),
            cumulative_reward=traces[-1].cumulative_reward,
            controller_total_seconds=sum(controller_times),
            controller_mean_seconds=fmean(controller_times),
            controller_p95_seconds=_percentile(controller_times, 95.0) or 0.0,
            online_update_seconds=sum(trace.adaptation_seconds for trace in traces),
            online_updates=sum(trace.adaptation_updates for trace in traces),
            offline_training_seconds=offline_training_seconds,
            recovery_time_slots=_recovery_time(traces, change_slot),
            adaptation_curve_reward=adaptation_curve,
        ),
        tuple(outcomes),
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    centered = x_array - x_array.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, y_array - y_array.mean()) / denominator)


def _adaptation_curve(
    traces: Sequence[SlotTrace], change_slot: int | None
) -> dict[str, float]:
    if change_slot is None:
        return {}
    result: dict[str, float] = {}
    for offset in (0, 1, 2, 5, 10, 20):
        values = [
            trace.reward
            for trace in traces
            if change_slot <= trace.slot <= change_slot + offset
        ]
        if values:
            result[str(offset)] = fmean(values)
    return result


def _recovery_time(
    traces: Sequence[SlotTrace], change_slot: int | None
) -> int | None:
    if change_slot is None or change_slot <= 1:
        return None
    before = [trace for trace in traces if trace.slot < change_slot]
    after = [trace for trace in traces if trace.slot >= change_slot]
    if not before or not after:
        return None
    window = max(2, min(10, len(before)))
    baseline = fmean(trace.total_backlog_cycles for trace in before[-window:])
    tolerance = max(baseline * 1.10, 1.0)
    for index in range(0, len(after) - window + 1):
        mean_backlog = fmean(
            trace.total_backlog_cycles for trace in after[index : index + window]
        )
        if mean_backlog <= tolerance:
            return after[index + window - 1].slot - change_slot + 1
    return None
