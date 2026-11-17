"""Persistent queue, cloud-split, energy, and flight simulator for VEC."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from math import isclose, isfinite, sqrt
from typing import Iterable, Mapping, Sequence

from .control import (
    AgentObservation,
    ControlObservation,
    DestinationMask,
    DispatchTrace,
    ExecutedActionTrace,
)
from .mobility import (
    MobilityProvider,
    Position3D,
    SyntheticRoadMobility,
    VehicleRouteConfig,
)
from .energy import (
    BatteryConfig,
    BatteryState,
    BatteryTransition,
    EnergyBreakdown,
    RotaryWingEnergyConfig,
    SlotEnergyDemand,
    SlotEnergyLedger,
    SystemEnergyConfig,
    UAVEnergyModel,
)
from .queues import QueueKey, QueueStage, QueueUnit, WorkItem, WorkQueue
from .radio import LinkAllocation, LinkRequest, QueueAwareOFDMA, RadioConfig
from .return_plan import ReturnLandingPlanner, ReturnPlan
from .lyapunov import (
    LyapunovConfig,
    LyapunovScore,
    LyapunovSlotInput,
    LyapunovTracker,
)
from .safety import (
    FlightLimits,
    FlightSafetyController,
    MotionCommand,
    MotionRepairResult,
    UAVKinematicState,
    Velocity3D,
)
from .tasks import DestinationKind, TaskRecord, TaskSpec, TaskState


@dataclass(frozen=True)
class Destination:
    kind: DestinationKind
    node_id: int

    @classmethod
    def local(cls, vehicle_id: int) -> "Destination":
        return cls(DestinationKind.LOCAL, vehicle_id)

    @classmethod
    def rsu(cls, rsu_id: int) -> "Destination":
        return cls(DestinationKind.RSU, rsu_id)

    @classmethod
    def uav(cls, uav_id: int) -> "Destination":
        return cls(DestinationKind.UAV, uav_id)


@dataclass(frozen=True)
class DispatchDecision:
    vehicle_id: int
    task_id: int
    destination: Destination
    uav_fraction: float = 1.0


@dataclass(frozen=True)
class ScheduledArrival:
    slot: int
    vehicle_id: int
    input_bits: float
    cpu_cycles: float
    deadline: float
    task_id: int | None = None
    partitionable: bool = False


@dataclass(frozen=True)
class ScheduledHarvest:
    """Energy that is deposited after service in the specified slot."""

    slot: int
    uav_id: int
    energy_j: float


class UAVMode(str, Enum):
    ACTIVE = "active"
    RETURNING = "returning"
    LANDED = "landed"
    FAILED = "failed"


@dataclass(frozen=True)
class StepAction:
    dispatches: tuple[DispatchDecision, ...] = ()
    cpu_weights: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    uav_motion_commands: tuple[MotionCommand, ...] = ()
    uav_cpu_frequencies_hz: tuple[float, ...] = ()


@dataclass(frozen=True)
class VecEnvConfig:
    slot_duration: float = 1.0
    vehicle_cpu_hz: tuple[float, ...] = (0.5e9,)
    rsu_cpu_hz: tuple[float, ...] = (8.0e9,)
    uav_cpu_hz: tuple[float, ...] = (4.0e9,)
    rsu_uplink_bps: tuple[float, ...] = (10.0e6,)
    uav_uplink_bps: tuple[float, ...] = (8.0e6,)
    uav_backhaul_bps: float = 20.0e6
    cloud_cpu_hz: float = 20.0e9
    merge_delay_s: float = 0.0
    radio: RadioConfig | None = None
    rsu_bandwidth_hz: tuple[float, ...] = (40.0e6,)
    uav_bandwidth_hz: tuple[float, ...] = (8.0e6,)
    vehicle_static_positions: tuple[Position3D, ...] = ()
    vehicle_routes: tuple[VehicleRouteConfig | None, ...] = ()
    rsu_positions: tuple[Position3D, ...] = ()
    uav_positions: tuple[Position3D, ...] = ()
    uav_landing_positions: tuple[Position3D, ...] = ()
    uav_initial_velocities: tuple[Velocity3D, ...] = ()
    uav_batteries: tuple[BatteryConfig, ...] = ()
    uav_reserve_j: tuple[float, ...] = ()
    uav_energy: RotaryWingEnergyConfig = field(
        default_factory=RotaryWingEnergyConfig
    )
    system_energy: SystemEnergyConfig = field(default_factory=SystemEnergyConfig)
    flight_limits: FlightLimits | None = None
    landing_position_tolerance_m: float = 0.1
    landing_velocity_tolerance_mps: float = 0.1
    max_return_plan_slots: int = 10_000
    lyapunov: LyapunovConfig = field(default_factory=LyapunovConfig)
    position_reference_m: float = 1_000.0
    max_episode_slots: int | None = None
    max_dispatches_per_vehicle: int = 1
    conservation_rtol: float = 1e-9
    conservation_atol: float = 1e-5

    def __post_init__(self) -> None:
        if self.slot_duration <= 0:
            raise ValueError("slot_duration must be positive")
        if not self.vehicle_cpu_hz:
            raise ValueError("at least one vehicle is required")
        for name, values in (
            ("vehicle_cpu_hz", self.vehicle_cpu_hz),
            ("rsu_cpu_hz", self.rsu_cpu_hz),
            ("uav_cpu_hz", self.uav_cpu_hz),
            ("rsu_uplink_bps", self.rsu_uplink_bps),
            ("uav_uplink_bps", self.uav_uplink_bps),
        ):
            if any(value < 0 for value in values):
                raise ValueError(f"{name} entries must be non-negative")
        if len(self.rsu_cpu_hz) != len(self.rsu_uplink_bps):
            raise ValueError("each RSU needs one CPU and uplink capacity")
        if len(self.uav_cpu_hz) != len(self.uav_uplink_bps):
            raise ValueError("each UAV needs one CPU and uplink capacity")
        vehicle_count = len(self.vehicle_cpu_hz)
        if self.vehicle_static_positions and len(self.vehicle_static_positions) != vehicle_count:
            raise ValueError("vehicle_static_positions must match vehicle count")
        if self.vehicle_routes and len(self.vehicle_routes) != vehicle_count:
            raise ValueError("vehicle_routes must match vehicle count")
        if self.rsu_positions and len(self.rsu_positions) != len(self.rsu_cpu_hz):
            raise ValueError("rsu_positions must match RSU count")
        if self.uav_positions and len(self.uav_positions) != len(self.uav_cpu_hz):
            raise ValueError("uav_positions must match UAV count")
        if self.uav_landing_positions and len(self.uav_landing_positions) != len(
            self.uav_cpu_hz
        ):
            raise ValueError("uav_landing_positions must match UAV count")
        if self.uav_landing_positions and self.flight_limits is None:
            raise ValueError("uav_landing_positions require flight_limits")
        if self.uav_landing_positions and not self.uav_batteries:
            raise ValueError("verified landing reserves require UAV batteries")
        if self.uav_initial_velocities and len(self.uav_initial_velocities) != len(
            self.uav_cpu_hz
        ):
            raise ValueError("uav_initial_velocities must match UAV count")
        if self.uav_batteries and len(self.uav_batteries) != len(self.uav_cpu_hz):
            raise ValueError("uav_batteries must match UAV count")
        if self.uav_reserve_j and len(self.uav_reserve_j) != len(self.uav_cpu_hz):
            raise ValueError("uav_reserve_j must match UAV count")
        if self.uav_reserve_j and not self.uav_batteries:
            raise ValueError("uav_reserve_j requires physical UAV batteries")
        if any(not isfinite(value) or value < 0 for value in self.uav_reserve_j):
            raise ValueError("UAV reserves must be finite and non-negative")
        for index, battery in enumerate(self.uav_batteries):
            reserve = self.uav_reserve_j[index] if self.uav_reserve_j else 0.0
            if reserve > battery.capacity_j:
                raise ValueError("UAV reserve cannot exceed battery capacity")
        if self.radio is not None:
            if len(self.rsu_bandwidth_hz) != len(self.rsu_cpu_hz):
                raise ValueError("each RSU needs one geometry-aware bandwidth")
            if len(self.uav_bandwidth_hz) != len(self.uav_cpu_hz):
                raise ValueError("each UAV needs one geometry-aware bandwidth")
            if any(value < 0 for value in self.rsu_bandwidth_hz + self.uav_bandwidth_hz):
                raise ValueError("radio bandwidths must be non-negative")
        if self.max_dispatches_per_vehicle <= 0:
            raise ValueError("max_dispatches_per_vehicle must be positive")
        if self.uav_backhaul_bps < 0 or self.cloud_cpu_hz < 0:
            raise ValueError("backhaul and cloud capacities must be non-negative")
        if self.merge_delay_s < 0:
            raise ValueError("merge_delay_s must be non-negative")
        if self.landing_position_tolerance_m <= 0:
            raise ValueError("landing_position_tolerance_m must be positive")
        if self.landing_velocity_tolerance_mps < 0:
            raise ValueError("landing_velocity_tolerance_mps must be non-negative")
        if self.max_return_plan_slots <= 0:
            raise ValueError("max_return_plan_slots must be positive")
        if not isfinite(self.position_reference_m) or self.position_reference_m <= 0:
            raise ValueError("position_reference_m must be finite and positive")
        if self.max_episode_slots is not None and self.max_episode_slots <= 0:
            raise ValueError("max_episode_slots must be positive when provided")
        if self.uav_landing_positions and self.flight_limits is not None:
            limits = self.flight_limits
            for target in self.uav_landing_positions:
                if not (
                    limits.x_min_m <= target[0] <= limits.x_max_m
                    and limits.y_min_m <= target[1] <= limits.y_max_m
                    and isclose(target[2], limits.altitude_min_m)
                ):
                    raise ValueError(
                        "landing sites must be inside the map at minimum altitude"
                    )


@dataclass(frozen=True)
class StepResult:
    slot: int
    start_time: float
    end_time: float
    completed_task_ids: tuple[int, ...]
    new_deadline_miss_ids: tuple[int, ...]
    new_arrival_ids: tuple[int, ...]
    queue_before: Mapping[str, float]
    queue_after: Mapping[str, float]
    queue_departures: Mapping[str, float]
    queue_arrivals: Mapping[str, float]
    compute_service_cycles: float
    external_arrival_cycles: float
    conservation_residual: float
    vehicle_positions_start: tuple[Position3D, ...]
    vehicle_positions_end: tuple[Position3D, ...]
    radio_allocations: tuple[LinkAllocation, ...]
    uav_positions_start: tuple[Position3D, ...]
    uav_positions_end: tuple[Position3D, ...]
    motion_repair: MotionRepairResult
    executed_uav_cpu_frequencies_hz: tuple[float, ...]
    uav_energy_j: tuple[float, ...]
    uav_energy_breakdown: tuple[EnergyBreakdown, ...]
    battery_transitions: tuple[BatteryTransition, ...]
    physical_feasibility_failure: bool
    uav_modes_start: tuple[UAVMode, ...]
    uav_modes_end: tuple[UAVMode, ...]
    return_reserve_j: tuple[float, ...]
    return_plans: tuple[ReturnPlan | None, ...]
    energy: SlotEnergyLedger
    lyapunov: LyapunovScore
    observation_before: ControlObservation
    observation_after: ControlObservation
    masks_before: tuple[DestinationMask, ...]
    masks_after: tuple[DestinationMask, ...]
    action_trace: ExecutedActionTrace
    terminated: bool
    truncated: bool


@dataclass
class _UAVSlotResources:
    energy_remaining_j: list[float]
    energy_spent_j: list[float]
    service_enabled: list[bool]
    cpu_frequencies_hz: list[float]
    return_reserve_j: list[float]
    return_plans: list[ReturnPlan | None]
    propulsion_j: list[float]
    computation_j: list[float]
    reception_j: list[float]
    backhaul_j: list[float]
    control_j: list[float]


@dataclass
class _SystemEnergyAccumulator:
    vehicle_compute_j: list[float]
    vehicle_uplink_j: list[float]
    rsu_compute_j: list[float]
    rsu_reception_j: list[float]
    cloud_compute_j: float = 0.0
    cloud_reception_j: float = 0.0


class VecQueueEnv:
    """Persistent, causally staged queue network.

    Queue backlogs exposed by this class are in equivalent CPU cycles even
    when the physical queue stores bits.  Radio capacities remain in bits.
    """

    def __init__(
        self,
        config: VecEnvConfig | None = None,
        arrival_trace: Iterable[ScheduledArrival] = (),
        harvest_trace: Iterable[ScheduledHarvest] = (),
        mobility_provider: MobilityProvider | None = None,
    ) -> None:
        self.config = config or VecEnvConfig()
        self.mobility: MobilityProvider = mobility_provider or SyntheticRoadMobility(
            vehicle_count=len(self.config.vehicle_cpu_hz),
            routes=self.config.vehicle_routes,
            static_positions=self.config.vehicle_static_positions,
        )
        self.rsu_positions = self._node_positions(
            self.config.rsu_positions, len(self.config.rsu_cpu_hz), altitude=0.0
        )
        self.uav_positions = self._node_positions(
            self.config.uav_positions, len(self.config.uav_cpu_hz), altitude=100.0
        )
        self._initial_uav_positions = self.uav_positions
        self.uav_landing_positions = tuple(self.config.uav_landing_positions)
        self._initial_uav_velocities = self.config.uav_initial_velocities or tuple(
            (0.0, 0.0, 0.0) for _ in self.config.uav_cpu_hz
        )
        self.uav_states = tuple(
            UAVKinematicState(position, velocity)
            for position, velocity in zip(
                self._initial_uav_positions, self._initial_uav_velocities
            )
        )
        self.energy_model = UAVEnergyModel(self.config.uav_energy)
        self.lyapunov_tracker = LyapunovTracker(
            len(self.config.uav_cpu_hz), self.config.lyapunov
        )
        self.batteries = tuple(
            BatteryState(battery) for battery in self.config.uav_batteries
        )
        self.flight_safety = (
            FlightSafetyController(self.config.flight_limits)
            if self.config.flight_limits is not None
            else None
        )
        self.return_planner = (
            ReturnLandingPlanner(
                self.flight_safety,
                self.energy_model,
                self.config.slot_duration,
                position_tolerance_m=self.config.landing_position_tolerance_m,
                velocity_tolerance_mps=self.config.landing_velocity_tolerance_mps,
                max_slots=self.config.max_return_plan_slots,
            )
            if self.uav_landing_positions and self.flight_safety is not None
            else None
        )
        self.radio_scheduler = (
            QueueAwareOFDMA()
            if self.config.radio is None
            else QueueAwareOFDMA.from_config(self.config.radio)
        )
        self._arrival_trace: dict[int, list[ScheduledArrival]] = defaultdict(list)
        for arrival in arrival_trace:
            if arrival.slot < 0:
                raise ValueError("arrival slots must be non-negative")
            self._arrival_trace[arrival.slot].append(arrival)
        for arrivals in self._arrival_trace.values():
            arrivals.sort(key=lambda item: (
                item.vehicle_id,
                -1 if item.task_id is None else item.task_id,
            ))
        self._harvest_trace: dict[tuple[int, int], float] = defaultdict(float)
        for harvest in harvest_trace:
            if harvest.slot < 0:
                raise ValueError("harvest slots must be non-negative")
            if not 0 <= harvest.uav_id < len(self.config.uav_cpu_hz):
                raise ValueError(f"invalid harvest UAV {harvest.uav_id}")
            if not isfinite(harvest.energy_j) or harvest.energy_j < 0:
                raise ValueError("harvest energy must be finite and non-negative")
            self._harvest_trace[(harvest.slot, harvest.uav_id)] += harvest.energy_j
        self.queues: dict[QueueKey, WorkQueue] = {}
        self.tasks: dict[int, TaskRecord] = {}
        self.slot = 0
        self._next_task_id = 0
        self.vehicle_positions: tuple[Position3D, ...] = ()
        self._uav_service_enabled = [True for _ in self.config.uav_cpu_hz]
        self.uav_modes = [UAVMode.ACTIVE for _ in self.config.uav_cpu_hz]
        self._episode_done = False
        self._build_queues()
        self.reset()

    @property
    def time(self) -> float:
        return self.slot * self.config.slot_duration

    def reset(self) -> tuple[int, ...]:
        self.slot = 0
        self._next_task_id = 0
        self.tasks = {}
        self._build_queues()
        self.mobility.reset()
        self.vehicle_positions = self.mobility.positions()
        if len(self.vehicle_positions) != len(self.config.vehicle_cpu_hz):
            raise ValueError("mobility provider returned the wrong vehicle count")
        self.uav_positions = self._initial_uav_positions
        self.uav_states = tuple(
            UAVKinematicState(position, velocity)
            for position, velocity in zip(
                self._initial_uav_positions, self._initial_uav_velocities
            )
        )
        for battery in self.batteries:
            battery.reset()
        self.lyapunov_tracker.reset()
        self._uav_service_enabled = [True for _ in self.config.uav_cpu_hz]
        self.uav_modes = [UAVMode.ACTIVE for _ in self.config.uav_cpu_hz]
        self._episode_done = False
        return tuple(self._commit_scheduled_arrivals(0))

    def submit_task(
        self,
        vehicle_id: int,
        input_bits: float,
        cpu_cycles: float,
        deadline: float,
        *,
        task_id: int | None = None,
        partitionable: bool = False,
    ) -> int:
        """Place a task in ingress at the current slot boundary."""

        self._validate_vehicle(vehicle_id)
        assigned_id = self._allocate_task_id(task_id)
        spec = TaskSpec(
            task_id=assigned_id,
            vehicle_id=vehicle_id,
            input_bits=float(input_bits),
            cpu_cycles=float(cpu_cycles),
            arrival_time=self.time,
            deadline=float(deadline),
            partitionable=partitionable,
        )
        record = TaskRecord(spec=spec)
        record.record_stage(self.slot, self._ingress_key(vehicle_id).label)
        self.tasks[assigned_id] = record
        self.queues[self._ingress_key(vehicle_id)].push(
            WorkItem(
                task_id=assigned_id,
                amount=spec.cpu_cycles,
                unit=QueueUnit.CYCLES,
                intensity=spec.intensity,
            )
        )
        return assigned_id

    def step(self, action: StepAction | None = None) -> StepResult:
        if self._episode_done:
            raise RuntimeError("episode is done; call reset before stepping again")
        action = action or StepAction()
        start_slot = self.slot
        start_time = self.time
        end_time = start_time + self.config.slot_duration
        positions_start = self.vehicle_positions
        uav_positions_start = self.uav_positions
        uav_modes_start = tuple(self.uav_modes)
        queue_before = self.queue_backlogs()
        unfinished_start = sum(
            not record.is_complete for record in self.tasks.values()
        )
        observation_before = self.control_observation()
        masks_before = self.destination_masks()
        departures: dict[str, float] = defaultdict(float)
        arrivals: dict[str, float] = defaultdict(float)
        system_energy = _SystemEnergyAccumulator(
            vehicle_compute_j=[0.0 for _ in self.config.vehicle_cpu_hz],
            vehicle_uplink_j=[0.0 for _ in self.config.vehicle_cpu_hz],
            rsu_compute_j=[0.0 for _ in self.config.rsu_cpu_hz],
            rsu_reception_j=[0.0 for _ in self.config.rsu_cpu_hz],
        )

        motion_repair, resources, physical_failure = self._prepare_uav_slot(action)
        self._uav_service_enabled = resources.service_enabled
        executed_dispatches = self._repair_dispatches_for_forced_return(
            action.dispatches,
            masks_before,
            resources.service_enabled,
        )
        self._apply_dispatches(executed_dispatches, departures, arrivals)
        completed_branches, compute_service = self._serve_compute(
            action.cpu_weights, departures, resources, system_energy
        )
        radio_allocations = self._serve_uploads(
            departures, arrivals, resources, system_energy
        )
        self._serve_backhaul(
            departures, arrivals, resources, system_energy
        )
        self._promote_received_tasks(departures, arrivals)
        self._promote_cloud_received(departures, arrivals)

        completed = self._commit_branch_completions(
            completed_branches, end_time, start_slot + 1
        )
        completed.extend(self._complete_ready_merges(end_time, start_slot + 1))
        completed = sorted(set(completed))

        positions_end = self.mobility.advance(self.config.slot_duration)
        if len(positions_end) != len(self.config.vehicle_cpu_hz):
            raise ValueError("mobility provider returned the wrong vehicle count")
        self.vehicle_positions = positions_end
        self.uav_positions = motion_repair.end_positions
        self.uav_states = tuple(
            UAVKinematicState(position, velocity)
            for position, velocity in zip(
                motion_repair.end_positions, motion_repair.executed_velocities
            )
        )
        self._update_uav_modes()
        battery_transitions = self._apply_battery_transitions(
            start_slot, resources.energy_spent_j
        )
        self.slot += 1
        new_arrivals = self._commit_scheduled_arrivals(self.slot)
        external_arrival_cycles = sum(
            self.tasks[task_id].spec.cpu_cycles for task_id in new_arrivals
        )
        for task_id in new_arrivals:
            arrivals[self._ingress_key(self.tasks[task_id].spec.vehicle_id).label] += (
                self.tasks[task_id].spec.cpu_cycles
            )

        new_misses = self._record_deadline_crossings(end_time)
        queue_after = self.queue_backlogs()
        before_total = sum(queue_before.values())
        after_total = sum(queue_after.values())
        expected_after_total = (
            before_total + external_arrival_cycles - compute_service
        )
        residual = after_total - expected_after_total
        if not isclose(
            after_total,
            expected_after_total,
            rel_tol=self.config.conservation_rtol,
            abs_tol=self.config.conservation_atol,
        ):
            raise AssertionError(
                "workload conservation failed: "
                f"before={before_total}, arrivals={external_arrival_cycles}, "
                f"compute={compute_service}, after={after_total}, "
                f"residual={residual}"
            )

        energy_ledger = SlotEnergyLedger(
            vehicle_compute_j=tuple(system_energy.vehicle_compute_j),
            vehicle_uplink_j=tuple(system_energy.vehicle_uplink_j),
            rsu_compute_j=tuple(system_energy.rsu_compute_j),
            rsu_reception_j=tuple(system_energy.rsu_reception_j),
            uav_j=tuple(resources.energy_spent_j),
            cloud_compute_j=system_energy.cloud_compute_j,
            cloud_reception_j=system_energy.cloud_reception_j,
        )
        harvest_j = tuple(
            self._harvest_trace.get((start_slot, uav_id), 0.0)
            for uav_id in range(len(self.config.uav_cpu_hz))
        )
        lyapunov = self.lyapunov_tracker.record(
            LyapunovSlotInput(
                queue_before=queue_before,
                queue_after=queue_after,
                queue_arrivals=dict(arrivals),
                queue_departures=dict(departures),
                uav_energy_j=tuple(resources.energy_spent_j),
                harvest_j=harvest_j,
                system_energy_j=energy_ledger.total_j,
                unfinished_tasks=unfinished_start,
                new_deadline_misses=len(new_misses),
                slot_duration_s=self.config.slot_duration,
                external_arrival_cycles=external_arrival_cycles,
                compute_service_cycles=compute_service,
            )
        )
        terminated = physical_failure
        truncated = (
            self.config.max_episode_slots is not None
            and self.slot >= self.config.max_episode_slots
            and not terminated
        )
        self._episode_done = terminated or truncated
        observation_after = self.control_observation()
        masks_after = self.destination_masks()
        proposed_velocities = (
            tuple(command.velocity for command in action.uav_motion_commands)
            if action.uav_motion_commands
            else tuple((0.0, 0.0, 0.0) for _ in self.config.uav_cpu_hz)
        )
        requested_frequencies = (
            action.uav_cpu_frequencies_hz
            if action.uav_cpu_frequencies_hz
            else self.config.uav_cpu_hz
        )
        action_trace = ExecutedActionTrace(
            dispatches=tuple(
                DispatchTrace(
                    vehicle_id=decision.vehicle_id,
                    task_id=decision.task_id,
                    destination_kind=decision.destination.kind.value,
                    destination_id=decision.destination.node_id,
                    uav_fraction=decision.uav_fraction,
                )
                for decision in executed_dispatches
            ),
            proposed_uav_velocities=proposed_velocities,
            executed_uav_velocities=motion_repair.executed_velocities,
            requested_uav_cpu_hz=tuple(requested_frequencies),
            executed_uav_cpu_hz=tuple(resources.cpu_frequencies_hz),
            motion_correction_mps=motion_repair.correction_magnitudes_mps,
            motion_candidates_checked=motion_repair.candidates_checked,
            physical_feasibility_failure=physical_failure,
        )

        return StepResult(
            slot=start_slot,
            start_time=start_time,
            end_time=end_time,
            completed_task_ids=tuple(completed),
            new_deadline_miss_ids=tuple(new_misses),
            new_arrival_ids=tuple(new_arrivals),
            queue_before=queue_before,
            queue_after=queue_after,
            queue_departures=dict(departures),
            queue_arrivals=dict(arrivals),
            compute_service_cycles=compute_service,
            external_arrival_cycles=external_arrival_cycles,
            conservation_residual=residual,
            vehicle_positions_start=positions_start,
            vehicle_positions_end=positions_end,
            radio_allocations=radio_allocations,
            uav_positions_start=uav_positions_start,
            uav_positions_end=self.uav_positions,
            motion_repair=motion_repair,
            executed_uav_cpu_frequencies_hz=tuple(
                resources.cpu_frequencies_hz
            ),
            uav_energy_j=tuple(resources.energy_spent_j),
            uav_energy_breakdown=tuple(
                EnergyBreakdown(
                    propulsion_j=resources.propulsion_j[uav_id],
                    computation_j=resources.computation_j[uav_id],
                    reception_j=resources.reception_j[uav_id],
                    backhaul_j=resources.backhaul_j[uav_id],
                    control_j=resources.control_j[uav_id],
                )
                for uav_id in range(len(resources.energy_spent_j))
            ),
            battery_transitions=battery_transitions,
            physical_feasibility_failure=physical_failure,
            uav_modes_start=uav_modes_start,
            uav_modes_end=tuple(self.uav_modes),
            return_reserve_j=tuple(resources.return_reserve_j),
            return_plans=tuple(resources.return_plans),
            energy=energy_ledger,
            lyapunov=lyapunov,
            observation_before=observation_before,
            observation_after=observation_after,
            masks_before=masks_before,
            masks_after=masks_after,
            action_trace=action_trace,
            terminated=terminated,
            truncated=truncated,
        )

    def queue_backlogs(self) -> dict[str, float]:
        return {
            key.label: queue.equivalent_backlog
            for key, queue in sorted(self.queues.items())
        }

    @property
    def total_backlog_cycles(self) -> float:
        return sum(queue.equivalent_backlog for queue in self.queues.values())

    def get_queue(self, stage: QueueStage, node_id: int) -> WorkQueue:
        return self.queues[QueueKey(stage, node_id)]

    def destination_masks(self) -> tuple[DestinationMask, ...]:
        masks: list[DestinationMask] = []
        uav_available = tuple(
            self._preview_uav_service_available(uav_id)
            for uav_id in range(len(self.config.uav_cpu_hz))
        )
        for vehicle_id in range(len(self.config.vehicle_cpu_hz)):
            position = self.vehicle_positions[vehicle_id]
            rsu_mask = tuple(
                self.config.radio is None
                or self.radio_scheduler.channel.in_rsu_coverage(
                    position, self.rsu_positions[rsu_id]
                )
                for rsu_id in range(len(self.config.rsu_cpu_hz))
            )
            uav_mask = tuple(
                uav_available[uav_id]
                and (
                    self.config.radio is None
                    or self.radio_scheduler.channel.in_uav_coverage(
                        position, self.uav_positions[uav_id]
                    )
                )
                for uav_id in range(len(self.config.uav_cpu_hz))
            )
            masks.append(
                DestinationMask(
                    vehicle_id=vehicle_id,
                    hold=True,
                    local=True,
                    rsu=rsu_mask,
                    uav=uav_mask,
                    ingress_task_ids=self.queues[
                        self._ingress_key(vehicle_id)
                    ].task_ids(),
                )
            )
        return tuple(masks)

    def control_observation(self) -> ControlObservation:
        qref = self.config.lyapunov.queue_reference_cycles
        pref = self.config.position_reference_m
        masks = self.destination_masks()
        vehicles: list[AgentObservation] = []
        for vehicle_id, mask in enumerate(masks):
            position = self.vehicle_positions[vehicle_id]
            vehicles.append(
                AgentObservation(
                    role="vehicle",
                    agent_id=vehicle_id,
                    features={
                        "ingress_q": self.queues[
                            self._ingress_key(vehicle_id)
                        ].equivalent_backlog
                        / qref,
                        "local_compute_q": self.queues[
                            QueueKey(QueueStage.COMPUTE_LOCAL, vehicle_id)
                        ].equivalent_backlog
                        / qref,
                        "x": position[0] / pref,
                        "y": position[1] / pref,
                        "z": position[2] / pref,
                        "reachable_rsu_fraction": (
                            sum(mask.rsu) / len(mask.rsu) if mask.rsu else 0.0
                        ),
                        "reachable_uav_fraction": (
                            sum(mask.uav) / len(mask.uav) if mask.uav else 0.0
                        ),
                    },
                    eligible_task_ids=mask.ingress_task_ids,
                )
            )

        rsus: list[AgentObservation] = []
        for rsu_id, position in enumerate(self.rsu_positions):
            rsus.append(
                AgentObservation(
                    role="rsu",
                    agent_id=rsu_id,
                    features={
                        "upload_q": self.queues[
                            QueueKey(QueueStage.UPLOAD_RSU, rsu_id)
                        ].equivalent_backlog
                        / qref,
                        "receive_q": self.queues[
                            QueueKey(QueueStage.RECEIVE_RSU, rsu_id)
                        ].equivalent_backlog
                        / qref,
                        "compute_q": self.queues[
                            QueueKey(QueueStage.COMPUTE_RSU, rsu_id)
                        ].equivalent_backlog
                        / qref,
                        "cpu_capacity": self.config.rsu_cpu_hz[rsu_id]
                        * self.config.slot_duration
                        / qref,
                        "x": position[0] / pref,
                        "y": position[1] / pref,
                        "z": position[2] / pref,
                    },
                    eligible_task_ids=self.queues[
                        QueueKey(QueueStage.COMPUTE_RSU, rsu_id)
                    ].task_ids(),
                )
            )

        normalized_virtual = self.lyapunov_tracker.normalized_virtual_energy()
        uavs: list[AgentObservation] = []
        for uav_id, state in enumerate(self.uav_states):
            battery_fraction = (
                self.batteries[uav_id].charge_j
                / self.batteries[uav_id].config.capacity_j
                if self.batteries
                else 1.0
            )
            features = {
                "upload_q": self.queues[
                    QueueKey(QueueStage.UPLOAD_UAV, uav_id)
                ].equivalent_backlog
                / qref,
                "receive_q": self.queues[
                    QueueKey(QueueStage.RECEIVE_UAV, uav_id)
                ].equivalent_backlog
                / qref,
                "compute_q": self.queues[
                    QueueKey(QueueStage.COMPUTE_UAV, uav_id)
                ].equivalent_backlog
                / qref,
                "backhaul_q": self.queues[
                    QueueKey(QueueStage.BACKHAUL_UAV, uav_id)
                ].equivalent_backlog
                / qref,
                "battery_fraction": battery_fraction,
                "virtual_energy_q": normalized_virtual[uav_id],
                "x": state.position[0] / pref,
                "y": state.position[1] / pref,
                "z": state.position[2] / pref,
                "vx": state.velocity[0] * self.config.slot_duration / pref,
                "vy": state.velocity[1] * self.config.slot_duration / pref,
                "vz": state.velocity[2] * self.config.slot_duration / pref,
                "active": float(self.uav_modes[uav_id] is UAVMode.ACTIVE),
                "returning": float(
                    self.uav_modes[uav_id] is UAVMode.RETURNING
                ),
                "landed": float(self.uav_modes[uav_id] is UAVMode.LANDED),
                "failed": float(self.uav_modes[uav_id] is UAVMode.FAILED),
            }
            task_ids = set(
                self.queues[
                    QueueKey(QueueStage.COMPUTE_UAV, uav_id)
                ].task_ids()
            )
            task_ids.update(
                self.queues[
                    QueueKey(QueueStage.BACKHAUL_UAV, uav_id)
                ].task_ids()
            )
            uavs.append(
                AgentObservation(
                    role="uav",
                    agent_id=uav_id,
                    features=features,
                    eligible_task_ids=tuple(sorted(task_ids)),
                )
            )

        global_names: list[str] = []
        global_values: list[float] = []
        for label, backlog in sorted(self.queue_backlogs().items()):
            global_names.append(f"queue:{label}")
            global_values.append(backlog / qref)
        for uav_id, value in enumerate(normalized_virtual):
            global_names.append(f"virtual_energy:{uav_id}")
            global_values.append(value)
        for uav_id, observation in enumerate(uavs):
            for feature in ("battery_fraction", "x", "y", "z"):
                global_names.append(f"uav:{uav_id}:{feature}")
                global_values.append(observation.features[feature])
        global_names.append("unfinished_tasks")
        global_values.append(
            float(sum(not record.is_complete for record in self.tasks.values()))
        )
        return ControlObservation(
            vehicles=tuple(vehicles),
            rsus=tuple(rsus),
            uavs=tuple(uavs),
            global_critic_state=tuple(global_values),
            global_feature_names=tuple(global_names),
        )

    def _preview_uav_service_available(self, uav_id: int) -> bool:
        if self.uav_modes[uav_id] is not UAVMode.ACTIVE:
            return False
        if not self.batteries:
            return True
        state = self.uav_states[uav_id]
        if self.flight_safety is not None:
            repaired = self.flight_safety.repair(
                (state,),
                (MotionCommand((0.0, 0.0, 0.0)),),
                self.config.slot_duration,
            )
            if not repaired.feasible:
                return False
            velocity = repaired.executed_velocities[0]
            end_state = UAVKinematicState(
                repaired.end_positions[0], velocity
            )
        else:
            velocity = (0.0, 0.0, 0.0)
            end_state = UAVKinematicState(state.position, velocity)
        baseline = self._baseline_uav_energy((velocity,))[0]
        reserve, plan = self._return_requirement(uav_id, end_state)
        if plan is not None and not plan.feasible:
            return False
        return (
            self.batteries[uav_id].charge_j - baseline - reserve > 1e-9
        )

    def _prepare_uav_slot(
        self, action: StepAction
    ) -> tuple[MotionRepairResult, _UAVSlotResources, bool]:
        """Repair motion and reserve a complete feasible landing trajectory."""

        count = len(self.config.uav_cpu_hz)
        if action.uav_motion_commands and len(action.uav_motion_commands) != count:
            raise ValueError("uav_motion_commands must match UAV count")
        commands = action.uav_motion_commands or tuple(
            MotionCommand((0.0, 0.0, 0.0)) for _ in range(count)
        )
        for command in commands:
            if len(command.velocity) != 3 or any(
                not isfinite(value) for value in command.velocity
            ):
                raise ValueError("UAV motion velocities must be finite 3-D vectors")

        if action.uav_cpu_frequencies_hz and len(
            action.uav_cpu_frequencies_hz
        ) != count:
            raise ValueError("uav_cpu_frequencies_hz must match UAV count")
        requested_frequencies = action.uav_cpu_frequencies_hz or self.config.uav_cpu_hz
        frequencies: list[float] = []
        for requested, maximum in zip(
            requested_frequencies, self.config.uav_cpu_hz
        ):
            if not isfinite(requested) or requested < 0:
                raise ValueError("UAV CPU frequencies must be finite and non-negative")
            frequencies.append(min(float(requested), maximum))

        proposed = tuple(command.velocity for command in commands)
        adjusted_commands = list(commands)
        forced_contingency: set[int] = set()
        candidates_checked = 0
        plans: list[ReturnPlan | None] = [None for _ in range(count)]
        reserves = [0.0 for _ in range(count)]
        physical_failure = False

        for uav_id, mode in enumerate(self.uav_modes):
            if mode in (UAVMode.LANDED, UAVMode.FAILED):
                adjusted_commands[uav_id] = MotionCommand((0.0, 0.0, 0.0))
                forced_contingency.add(uav_id)
            elif mode is UAVMode.RETURNING:
                plan = self._plan_return(uav_id, self.uav_states[uav_id])
                plans[uav_id] = plan
                forced_contingency.add(uav_id)
                if plan is None or not plan.feasible or plan.first_command is None:
                    physical_failure = plan is None or not plan.feasible
                    adjusted_commands[uav_id] = MotionCommand((0.0, 0.0, 0.0))
                else:
                    adjusted_commands[uav_id] = plan.first_command

        repair = self._repair_motion(tuple(adjusted_commands))
        candidates_checked += repair.candidates_checked

        # Reject normal motion that would leave too little energy for the full
        # verified return. Those UAVs immediately execute the first return
        # segment and are removed from service destinations.
        if self.batteries and repair.feasible and not physical_failure:
            for _ in range(count + 1):
                baseline = self._baseline_uav_energy(
                    repair.executed_velocities, self.uav_modes
                )
                newly_forced: set[int] = set()
                for uav_id in range(count):
                    if self.uav_modes[uav_id] is not UAVMode.ACTIVE:
                        continue
                    if uav_id in forced_contingency:
                        continue
                    end_state = UAVKinematicState(
                        repair.end_positions[uav_id],
                        repair.executed_velocities[uav_id],
                    )
                    reserve, plan = self._return_requirement(uav_id, end_state)
                    reserves[uav_id], plans[uav_id] = reserve, plan
                    if (
                        plan is not None
                        and not plan.feasible
                        or not self.batteries[uav_id].can_execute(
                            baseline[uav_id], reserve
                        )
                    ):
                        newly_forced.add(uav_id)
                if not newly_forced:
                    break
                for uav_id in newly_forced:
                    current_plan = self._plan_return(
                        uav_id, self.uav_states[uav_id]
                    )
                    plans[uav_id] = current_plan
                    forced_contingency.add(uav_id)
                    if self.return_planner is None:
                        adjusted_commands[uav_id] = MotionCommand(
                            (0.0, 0.0, 0.0)
                        )
                    elif (
                        current_plan is None
                        or not current_plan.feasible
                        or current_plan.first_command is None
                    ):
                        physical_failure = True
                    else:
                        self.uav_modes[uav_id] = UAVMode.RETURNING
                        adjusted_commands[uav_id] = current_plan.first_command
                if physical_failure:
                    break
                repair = self._repair_motion(tuple(adjusted_commands))
                candidates_checked += repair.candidates_checked
                if not repair.feasible:
                    break

        physical_failure = physical_failure or not repair.feasible
        baseline = self._baseline_uav_energy(
            repair.executed_velocities, self.uav_modes
        )
        if self.batteries and repair.feasible and not physical_failure:
            contingency_states: list[UAVKinematicState] = []
            for uav_id in range(count):
                mode = self.uav_modes[uav_id]
                end_state = UAVKinematicState(
                    repair.end_positions[uav_id],
                    repair.executed_velocities[uav_id],
                )
                contingency_states.append(end_state)
                if mode is UAVMode.LANDED:
                    reserves[uav_id] = 0.0
                    plans[uav_id] = self._plan_return(uav_id, end_state)
                    continue
                reserve, plan = self._return_requirement(uav_id, end_state)
                reserves[uav_id], plans[uav_id] = reserve, plan
                if plan is not None and not plan.feasible:
                    physical_failure = True
                    break
                if mode is UAVMode.RETURNING:
                    if not self.batteries[uav_id].can_execute(
                        baseline[uav_id], reserve
                    ):
                        physical_failure = True
                        break
                elif mode is UAVMode.ACTIVE and not self.batteries[
                    uav_id
                ].can_execute(baseline[uav_id], reserve):
                    physical_failure = True
                    break
            if (
                not physical_failure
                and self.return_planner is not None
                and not self._joint_return_is_safe(contingency_states, plans)
            ):
                physical_failure = True

        if physical_failure:
            # An infeasible result is explicit and is never accepted as a safe
            # trajectory. Holding position is the least-action failure state.
            zeros = tuple((0.0, 0.0, 0.0) for _ in range(count))
            ends = tuple(state.position for state in self.uav_states)
            minimum = FlightSafetyController.minimum_pair_separation(
                self.uav_states, zeros, self.config.slot_duration
            )
            repair = MotionRepairResult(
                proposed_velocities=proposed,
                executed_velocities=zeros,
                end_positions=ends,
                repaired=tuple(True for _ in range(count)),
                minimum_separation_m=minimum,
                candidates_checked=candidates_checked,
                feasible=False,
            )
            baseline = self._baseline_uav_energy(zeros, self.uav_modes)
            for uav_id, mode in enumerate(self.uav_modes):
                if mode is not UAVMode.LANDED:
                    self.uav_modes[uav_id] = UAVMode.FAILED

        baseline_components = self._baseline_uav_energy_components(
            repair.executed_velocities, self.uav_modes
        )
        baseline = [component.total_j for component in baseline_components]
        spent: list[float] = []
        remaining: list[float] = []
        enabled: list[bool] = []
        for uav_id, expenditure in enumerate(baseline):
            if not self.batteries:
                spend = expenditure
                can_serve = repair.feasible
                budget = float("inf") if can_serve else 0.0
            else:
                battery = self.batteries[uav_id]
                affordable = expenditure <= battery.charge_j + 1e-9
                spend = min(expenditure, battery.charge_j) if affordable else 0.0
                can_serve = (
                    repair.feasible
                    and self.uav_modes[uav_id] is UAVMode.ACTIVE
                    and uav_id not in forced_contingency
                    and battery.can_execute(
                        expenditure, reserves[uav_id]
                    )
                )
                budget = (
                    max(
                        battery.charge_j
                        - expenditure
                        - reserves[uav_id],
                        0.0,
                    )
                    if can_serve
                    else 0.0
                )
            spent.append(spend)
            remaining.append(budget)
            enabled.append(can_serve)
            if not can_serve:
                frequencies[uav_id] = 0.0

        repair = MotionRepairResult(
            proposed_velocities=proposed,
            executed_velocities=repair.executed_velocities,
            end_positions=repair.end_positions,
            repaired=tuple(
                any(abs(a - b) > 1e-9 for a, b in zip(executed, requested))
                for executed, requested in zip(
                    repair.executed_velocities, proposed
                )
            ),
            minimum_separation_m=repair.minimum_separation_m,
            candidates_checked=candidates_checked,
            feasible=repair.feasible,
        )
        baseline_scales = [
            spent[index] / baseline[index] if baseline[index] > 0.0 else 0.0
            for index in range(count)
        ]
        return (
            repair,
            _UAVSlotResources(
                remaining,
                spent,
                enabled,
                frequencies,
                reserves,
                plans,
                [
                    item.propulsion_j * baseline_scales[index]
                    for index, item in enumerate(baseline_components)
                ],
                [0.0 for _ in range(count)],
                [0.0 for _ in range(count)],
                [0.0 for _ in range(count)],
                [
                    item.control_j * baseline_scales[index]
                    for index, item in enumerate(baseline_components)
                ],
            ),
            physical_failure,
        )

    def _repair_motion(
        self, commands: tuple[MotionCommand, ...]
    ) -> MotionRepairResult:
        if self.flight_safety is not None:
            return self.flight_safety.repair(
                self.uav_states,
                commands,
                self.config.slot_duration,
                allow_min_altitude_arrival=tuple(
                    mode is UAVMode.RETURNING for mode in self.uav_modes
                ),
            )
        velocities = tuple(command.velocity for command in commands)
        ends = tuple(
            tuple(
                coordinate + self.config.slot_duration * speed
                for coordinate, speed in zip(state.position, velocity)
            )
            for state, velocity in zip(self.uav_states, velocities)
        )
        return MotionRepairResult(
            proposed_velocities=velocities,
            executed_velocities=velocities,
            end_positions=ends,  # type: ignore[arg-type]
            repaired=tuple(False for _ in velocities),
            minimum_separation_m=FlightSafetyController.minimum_pair_separation(
                self.uav_states, velocities, self.config.slot_duration
            ),
            candidates_checked=1,
            feasible=True,
        )

    def _baseline_uav_energy(
        self,
        velocities: Sequence[Velocity3D],
        modes: Sequence[UAVMode] | None = None,
    ) -> list[float]:
        return [
            item.total_j
            for item in self._baseline_uav_energy_components(velocities, modes)
        ]

    def _baseline_uav_energy_components(
        self,
        velocities: Sequence[Velocity3D],
        modes: Sequence[UAVMode] | None = None,
    ) -> list[EnergyBreakdown]:
        duration = self.config.slot_duration
        active_modes = modes or tuple(UAVMode.ACTIVE for _ in velocities)
        return [
            self.energy_model.slot_energy(
                SlotEnergyDemand(
                    horizontal_speed_mps=sqrt(velocity[0] ** 2 + velocity[1] ** 2),
                    vertical_speed_mps=velocity[2],
                    control_active_seconds=(
                        0.0 if mode is UAVMode.LANDED else duration
                    ),
                    airborne=mode is not UAVMode.LANDED,
                ),
                duration,
            )
            for velocity, mode in zip(velocities, active_modes)
        ]

    def _plan_return(
        self, uav_id: int, state: UAVKinematicState
    ) -> ReturnPlan | None:
        if self.return_planner is None:
            return None
        return self.return_planner.plan(
            state, self.uav_landing_positions[uav_id]
        )

    def _return_requirement(
        self, uav_id: int, state: UAVKinematicState
    ) -> tuple[float, ReturnPlan | None]:
        configured = self._uav_reserve(uav_id)
        plan = self._plan_return(uav_id, state)
        if plan is None:
            return configured, None
        return max(configured, plan.energy_j), plan

    def _joint_return_is_safe(
        self,
        starting_states: Sequence[UAVKinematicState],
        plans: Sequence[ReturnPlan | None],
    ) -> bool:
        """Verify synchronized contingency paths, including landed padding."""

        if self.flight_safety is None or len(starting_states) < 2:
            return True
        if any(plan is None or not plan.feasible for plan in plans):
            return False
        concrete = [plan for plan in plans if plan is not None]
        longest = max((plan.slots for plan in concrete), default=0)
        states = list(starting_states)
        for slot in range(longest):
            velocities: list[Velocity3D] = []
            for uav_id, plan in enumerate(concrete):
                velocity = (
                    plan.executed_velocities[slot]
                    if slot < plan.slots
                    else (0.0, 0.0, 0.0)
                )
                velocities.append(velocity)
            minimum = FlightSafetyController.minimum_pair_separation(
                states, velocities, self.config.slot_duration
            )
            if minimum + 1e-9 < self.flight_safety.limits.minimum_separation_m:
                return False
            states = [
                UAVKinematicState(
                    (
                        plan.end_positions[slot]
                        if slot < plan.slots
                        else plan.target_position
                    ),
                    velocity,
                )
                for plan, velocity in zip(concrete, velocities)
            ]
        return True

    def _update_uav_modes(self) -> None:
        if self.return_planner is None:
            return
        for uav_id, mode in enumerate(self.uav_modes):
            if mode is UAVMode.RETURNING and self.return_planner.is_landed(
                self.uav_states[uav_id], self.uav_landing_positions[uav_id]
            ):
                self.uav_modes[uav_id] = UAVMode.LANDED

    def _uav_reserve(self, uav_id: int) -> float:
        return self.config.uav_reserve_j[uav_id] if self.config.uav_reserve_j else 0.0

    def _vehicle_tx_power_w(self) -> float:
        if self.config.radio is not None:
            return self.config.radio.vehicle_tx_power_w
        return self.config.system_energy.vehicle_tx_power_w

    def _apply_battery_transitions(
        self, slot: int, expenditures: Sequence[float]
    ) -> tuple[BatteryTransition, ...]:
        return tuple(
            battery.apply(
                expenditures[uav_id], self._harvest_trace.get((slot, uav_id), 0.0)
            )
            for uav_id, battery in enumerate(self.batteries)
        )

    def _build_queues(self) -> None:
        queues: dict[QueueKey, WorkQueue] = {}
        for vehicle_id in range(len(self.config.vehicle_cpu_hz)):
            self._add_queue(queues, self._ingress_key(vehicle_id), QueueUnit.CYCLES)
            self._add_queue(
                queues,
                QueueKey(QueueStage.COMPUTE_LOCAL, vehicle_id),
                QueueUnit.CYCLES,
            )
        for rsu_id in range(len(self.config.rsu_cpu_hz)):
            self._add_queue(
                queues, QueueKey(QueueStage.UPLOAD_RSU, rsu_id), QueueUnit.BITS
            )
            self._add_queue(
                queues, QueueKey(QueueStage.RECEIVE_RSU, rsu_id), QueueUnit.BITS
            )
            self._add_queue(
                queues, QueueKey(QueueStage.COMPUTE_RSU, rsu_id), QueueUnit.CYCLES
            )
        for uav_id in range(len(self.config.uav_cpu_hz)):
            self._add_queue(
                queues, QueueKey(QueueStage.UPLOAD_UAV, uav_id), QueueUnit.BITS
            )
            self._add_queue(
                queues, QueueKey(QueueStage.RECEIVE_UAV, uav_id), QueueUnit.BITS
            )
            self._add_queue(
                queues, QueueKey(QueueStage.COMPUTE_UAV, uav_id), QueueUnit.CYCLES
            )
            self._add_queue(
                queues, QueueKey(QueueStage.BACKHAUL_UAV, uav_id), QueueUnit.BITS
            )
        self._add_queue(
            queues, QueueKey(QueueStage.RECEIVE_CLOUD, 0), QueueUnit.BITS
        )
        self._add_queue(
            queues, QueueKey(QueueStage.COMPUTE_CLOUD, 0), QueueUnit.CYCLES
        )
        self.queues = queues

    @staticmethod
    def _add_queue(
        queues: dict[QueueKey, WorkQueue], key: QueueKey, unit: QueueUnit
    ) -> None:
        queues[key] = WorkQueue(key, unit)

    @staticmethod
    def _ingress_key(vehicle_id: int) -> QueueKey:
        return QueueKey(QueueStage.INGRESS, vehicle_id)

    def _apply_dispatches(
        self,
        dispatches: Sequence[DispatchDecision],
        departures: dict[str, float],
        arrivals: dict[str, float],
    ) -> None:
        counts: dict[int, int] = defaultdict(int)
        seen_tasks: set[int] = set()
        for decision in dispatches:
            self._validate_vehicle(decision.vehicle_id)
            if decision.task_id in seen_tasks:
                raise ValueError(f"task {decision.task_id} was dispatched twice")
            seen_tasks.add(decision.task_id)
            counts[decision.vehicle_id] += 1
            if counts[decision.vehicle_id] > self.config.max_dispatches_per_vehicle:
                raise ValueError(
                    f"vehicle {decision.vehicle_id} exceeds its dispatch budget"
                )
            record = self._task(decision.task_id)
            if record.spec.vehicle_id != decision.vehicle_id:
                raise ValueError("a vehicle may dispatch only its own task")
            if record.state is not TaskState.INGRESS:
                raise ValueError(f"task {decision.task_id} is not in ingress")
            self._validate_destination(decision.destination, decision.vehicle_id)
            if not 0.0 <= decision.uav_fraction <= 1.0:
                raise ValueError("uav_fraction must lie in [0, 1]")
            if decision.destination.kind is not DestinationKind.UAV and not isclose(
                decision.uav_fraction, 1.0
            ):
                raise ValueError("uav_fraction applies only to UAV destinations")
            if (
                decision.destination.kind is DestinationKind.UAV
                and not record.spec.partitionable
                and not isclose(decision.uav_fraction, 1.0)
            ):
                raise ValueError("indivisible tasks require uav_fraction=1")

            source_key = self._ingress_key(decision.vehicle_id)
            source_item = self.queues[source_key].pop(decision.task_id)
            target_key, target_unit = self._dispatch_target(decision.destination)
            target_amount = (
                record.spec.cpu_cycles
                if target_unit is QueueUnit.CYCLES
                else record.spec.input_bits
            )
            self.queues[target_key].push(
                WorkItem(
                    task_id=decision.task_id,
                    amount=target_amount,
                    unit=target_unit,
                    intensity=record.spec.intensity,
                )
            )
            departures[source_key.label] += source_item.equivalent_cycles
            arrivals[target_key.label] += source_item.equivalent_cycles
            record.destination_kind = decision.destination.kind
            record.destination_id = decision.destination.node_id
            record.dispatch_time = self.time
            record.configure_branches(
                decision.uav_fraction
                if decision.destination.kind is DestinationKind.UAV
                else None
            )
            record.state = (
                TaskState.PROCESSING
                if decision.destination.kind is DestinationKind.LOCAL
                else TaskState.UPLOAD
            )
            record.record_stage(self.slot, target_key.label)

    @staticmethod
    def _repair_dispatches_for_forced_return(
        dispatches: Sequence[DispatchDecision],
        masks_before: Sequence[DestinationMask],
        service_enabled: Sequence[bool],
    ) -> tuple[DispatchDecision, ...]:
        """Hold work when a valid UAV becomes unavailable during slot repair.

        Destination masks are observed before joint motion and energy repair.
        A proposed movement can increase the verified landing reserve enough
        to force a UAV into return mode in the same slot.  A dispatch that was
        valid in the observed mask must then execute as hold, not crash the
        transition or send work to an unavailable server.  Dispatches that
        were already masked remain untouched here and are rejected by the
        normal destination validator.
        """

        observed = {mask.vehicle_id: mask for mask in masks_before}
        repaired: list[DispatchDecision] = []
        for decision in dispatches:
            if decision.destination.kind is not DestinationKind.UAV:
                repaired.append(decision)
                continue
            uav_id = decision.destination.node_id
            mask = observed.get(decision.vehicle_id)
            was_observed_feasible = (
                mask is not None
                and 0 <= uav_id < len(mask.uav)
                and mask.uav[uav_id]
            )
            became_unavailable = (
                0 <= uav_id < len(service_enabled)
                and not service_enabled[uav_id]
            )
            if was_observed_feasible and became_unavailable:
                continue
            repaired.append(decision)
        return tuple(repaired)

    def _serve_compute(
        self,
        all_weights: Mapping[str, Mapping[int, float]],
        departures: dict[str, float],
        resources: _UAVSlotResources,
        system_energy: _SystemEnergyAccumulator,
    ) -> tuple[list[tuple[int, str]], float]:
        completed: list[tuple[int, str]] = []
        service_total = 0.0
        compute_queues: list[tuple[QueueKey, float, int | None]] = []
        compute_queues.extend(
            (QueueKey(QueueStage.COMPUTE_LOCAL, node_id), frequency, None)
            for node_id, frequency in enumerate(self.config.vehicle_cpu_hz)
        )
        compute_queues.extend(
            (QueueKey(QueueStage.COMPUTE_RSU, node_id), frequency, None)
            for node_id, frequency in enumerate(self.config.rsu_cpu_hz)
        )
        compute_queues.extend(
            (
                QueueKey(QueueStage.COMPUTE_UAV, node_id),
                resources.cpu_frequencies_hz[node_id],
                node_id,
            )
            for node_id in range(len(self.config.uav_cpu_hz))
        )
        compute_queues.append(
            (QueueKey(QueueStage.COMPUTE_CLOUD, 0), self.config.cloud_cpu_hz, None)
        )

        for key, frequency, uav_id in compute_queues:
            queue = self.queues[key]
            capacity = frequency * self.config.slot_duration
            joules_per_cycle = 0.0
            if uav_id is not None:
                joules_per_cycle = (
                    self.config.uav_energy.compute_kappa * frequency**2
                )
                if joules_per_cycle > 0:
                    capacity = min(
                        capacity,
                        resources.energy_remaining_j[uav_id]
                        / joules_per_cycle,
                    )
            portions = queue.serve_weighted(
                capacity,
                all_weights.get(key.label),
            )
            served_here = sum(
                portion.equivalent_cycles for portion in portions
            )
            if uav_id is not None:
                energy = joules_per_cycle * served_here
                resources.energy_spent_j[uav_id] += energy
                resources.computation_j[uav_id] += energy
                resources.energy_remaining_j[uav_id] = max(
                    resources.energy_remaining_j[uav_id] - energy, 0.0
                )
            elif key.stage is QueueStage.COMPUTE_LOCAL:
                system_energy.vehicle_compute_j[key.node_id] += (
                    self.config.system_energy.vehicle_compute_kappa
                    * frequency**2
                    * served_here
                )
            elif key.stage is QueueStage.COMPUTE_RSU:
                system_energy.rsu_compute_j[key.node_id] += (
                    self.config.system_energy.rsu_compute_kappa
                    * frequency**2
                    * served_here
                )
            elif key.stage is QueueStage.COMPUTE_CLOUD:
                system_energy.cloud_compute_j += (
                    self.config.system_energy.cloud_compute_kappa
                    * frequency**2
                    * served_here
                )
            for portion in portions:
                departures[key.label] += portion.equivalent_cycles
                service_total += portion.equivalent_cycles
                if portion.completed_item:
                    completed.append((portion.task_id, portion.branch_id))
        completed.sort()
        return completed, service_total

    def _serve_uploads(
        self,
        departures: dict[str, float],
        arrivals: dict[str, float],
        resources: _UAVSlotResources,
        system_energy: _SystemEnergyAccumulator,
    ) -> tuple[LinkAllocation, ...]:
        if self.config.radio is not None:
            return self._serve_geometry_uploads(
                departures, arrivals, resources, system_energy
            )
        radio_queues: list[tuple[QueueKey, QueueKey, float, int | None]] = []
        radio_queues.extend(
            (
                QueueKey(QueueStage.UPLOAD_RSU, node_id),
                QueueKey(QueueStage.RECEIVE_RSU, node_id),
                capacity,
                None,
            )
            for node_id, capacity in enumerate(self.config.rsu_uplink_bps)
        )
        radio_queues.extend(
            (
                QueueKey(QueueStage.UPLOAD_UAV, node_id),
                QueueKey(QueueStage.RECEIVE_UAV, node_id),
                capacity,
                node_id,
            )
            for node_id, capacity in enumerate(self.config.uav_uplink_bps)
        )
        for source_key, staging_key, rate, uav_id in radio_queues:
            active_seconds = self.config.slot_duration
            if uav_id is not None:
                if not resources.service_enabled[uav_id]:
                    active_seconds = 0.0
                elif self.config.uav_energy.receive_power_w > 0:
                    active_seconds = min(
                        active_seconds,
                        resources.energy_remaining_j[uav_id]
                        / self.config.uav_energy.receive_power_w,
                    )
            portions = self.queues[source_key].serve_fifo(
                rate * active_seconds
            )
            if rate > 0:
                for portion in portions:
                    vehicle_id = self.tasks[portion.task_id].spec.vehicle_id
                    system_energy.vehicle_uplink_j[vehicle_id] += (
                        self._vehicle_tx_power_w() * portion.amount / rate
                    )
                if source_key.stage is QueueStage.UPLOAD_RSU:
                    system_energy.rsu_reception_j[source_key.node_id] += (
                        self.config.system_energy.rsu_receive_power_w
                        * min(
                            sum(portion.amount for portion in portions) / rate,
                            active_seconds,
                        )
                    )
            if uav_id is not None and rate > 0:
                served_bits = sum(portion.amount for portion in portions)
                receive_seconds = min(served_bits / rate, active_seconds)
                energy = self.config.uav_energy.receive_power_w * receive_seconds
                resources.energy_spent_j[uav_id] += energy
                resources.reception_j[uav_id] += energy
                resources.energy_remaining_j[uav_id] = max(
                    resources.energy_remaining_j[uav_id] - energy, 0.0
                )
            for portion in portions:
                record = self.tasks[portion.task_id]
                self.queues[staging_key].push(
                    WorkItem(
                        task_id=portion.task_id,
                        amount=portion.amount,
                        unit=QueueUnit.BITS,
                        intensity=record.spec.intensity,
                    )
                )
                departures[source_key.label] += portion.equivalent_cycles
                arrivals[staging_key.label] += portion.equivalent_cycles
        return ()

    def _serve_geometry_uploads(
        self,
        departures: dict[str, float],
        arrivals: dict[str, float],
        resources: _UAVSlotResources,
        system_energy: _SystemEnergyAccumulator,
    ) -> tuple[LinkAllocation, ...]:
        radio_queues: list[tuple[QueueKey, QueueKey, Position3D, float, bool]] = []
        radio_queues.extend(
            (
                QueueKey(QueueStage.UPLOAD_RSU, node_id),
                QueueKey(QueueStage.RECEIVE_RSU, node_id),
                self.rsu_positions[node_id],
                self.config.rsu_bandwidth_hz[node_id],
                False,
            )
            for node_id in range(len(self.config.rsu_cpu_hz))
        )
        radio_queues.extend(
            (
                QueueKey(QueueStage.UPLOAD_UAV, node_id),
                QueueKey(QueueStage.RECEIVE_UAV, node_id),
                self.uav_positions[node_id],
                self.config.uav_bandwidth_hz[node_id],
                True,
            )
            for node_id in range(len(self.config.uav_cpu_hz))
        )
        all_allocations: list[LinkAllocation] = []
        for source_key, staging_key, node_position, bandwidth, aerial in radio_queues:
            source = self.queues[source_key]
            requests: list[LinkRequest] = []
            for task_id in source.task_ids():
                record = self.tasks[task_id]
                item = source.get(task_id)
                if item is None:
                    continue
                requests.append(
                    LinkRequest(
                        task_id=task_id,
                        vehicle_id=record.spec.vehicle_id,
                        vehicle_position=self.vehicle_positions[record.spec.vehicle_id],
                        remaining_bits=item.amount,
                        equivalent_cycles=item.equivalent_cycles,
                        age_seconds=self.time - record.spec.arrival_time,
                        deadline_seconds=record.spec.deadline,
                    )
                )
            allocations = (
                self.radio_scheduler.schedule_uav(
                    requests,
                    node_position,
                    bandwidth,
                    self.config.slot_duration,
                )
                if aerial
                else self.radio_scheduler.schedule_rsu(
                    requests,
                    node_position,
                    bandwidth,
                    self.config.slot_duration,
                )
            )
            executed_allocations = allocations
            if aerial:
                uav_id = source_key.node_id
                planned_seconds = max(
                    (
                        allocation.bit_budget / allocation.rate_bps
                        if allocation.rate_bps > 0
                        else 0.0
                    )
                    for allocation in allocations
                ) if allocations else 0.0
                affordable_seconds = planned_seconds
                if not resources.service_enabled[uav_id]:
                    affordable_seconds = 0.0
                elif self.config.uav_energy.receive_power_w > 0:
                    affordable_seconds = min(
                        affordable_seconds,
                        resources.energy_remaining_j[uav_id]
                        / self.config.uav_energy.receive_power_w,
                    )
                scale = (
                    min(affordable_seconds / planned_seconds, 1.0)
                    if planned_seconds > 0
                    else 1.0
                )
                executed_allocations = tuple(
                    LinkAllocation(
                        task_id=allocation.task_id,
                        vehicle_id=allocation.vehicle_id,
                        bandwidth_hz=allocation.bandwidth_hz,
                        channel_gain=allocation.channel_gain,
                        rate_bps=allocation.rate_bps,
                        bit_budget=allocation.bit_budget * scale,
                        priority=allocation.priority,
                    )
                    for allocation in allocations
                )
            all_allocations.extend(executed_allocations)
            actual_active_seconds = 0.0
            for allocation in executed_allocations:
                portion = source.serve_task(
                    allocation.task_id, allocation.bit_budget
                )
                if portion is None:
                    continue
                record = self.tasks[portion.task_id]
                self.queues[staging_key].push(
                    WorkItem(
                        task_id=portion.task_id,
                        amount=portion.amount,
                        unit=QueueUnit.BITS,
                        intensity=record.spec.intensity,
                    )
                )
                departures[source_key.label] += portion.equivalent_cycles
                arrivals[staging_key.label] += portion.equivalent_cycles
                if allocation.rate_bps > 0:
                    system_energy.vehicle_uplink_j[allocation.vehicle_id] += (
                        self._vehicle_tx_power_w()
                        * portion.amount
                        / allocation.rate_bps
                    )
                if aerial and allocation.rate_bps > 0:
                    actual_active_seconds = max(
                        actual_active_seconds,
                        portion.amount / allocation.rate_bps,
                    )
                elif not aerial and allocation.rate_bps > 0:
                    actual_active_seconds = max(
                        actual_active_seconds,
                        portion.amount / allocation.rate_bps,
                    )
            if aerial:
                uav_id = source_key.node_id
                energy = (
                    self.config.uav_energy.receive_power_w
                    * actual_active_seconds
                )
                resources.energy_spent_j[uav_id] += energy
                resources.reception_j[uav_id] += energy
                resources.energy_remaining_j[uav_id] = max(
                    resources.energy_remaining_j[uav_id] - energy, 0.0
                )
            else:
                system_energy.rsu_reception_j[source_key.node_id] += (
                    self.config.system_energy.rsu_receive_power_w
                    * actual_active_seconds
                )
        return tuple(all_allocations)

    def _serve_backhaul(
        self,
        departures: dict[str, float],
        arrivals: dict[str, float],
        resources: _UAVSlotResources,
        system_energy: _SystemEnergyAccumulator,
    ) -> None:
        cloud_staging_key = QueueKey(QueueStage.RECEIVE_CLOUD, 0)
        for uav_id in range(len(self.config.uav_cpu_hz)):
            source_key = QueueKey(QueueStage.BACKHAUL_UAV, uav_id)
            active_seconds = self.config.slot_duration
            if not resources.service_enabled[uav_id]:
                active_seconds = 0.0
            elif self.config.uav_energy.backhaul_tx_power_w > 0:
                active_seconds = min(
                    active_seconds,
                    resources.energy_remaining_j[uav_id]
                    / self.config.uav_energy.backhaul_tx_power_w,
                )
            portions = self.queues[source_key].serve_fifo(
                self.config.uav_backhaul_bps * active_seconds
            )
            if self.config.uav_backhaul_bps > 0:
                served_bits = sum(portion.amount for portion in portions)
                actual_seconds = min(
                    served_bits / self.config.uav_backhaul_bps, active_seconds
                )
                energy = self.config.uav_energy.backhaul_tx_power_w * actual_seconds
                resources.energy_spent_j[uav_id] += energy
                resources.backhaul_j[uav_id] += energy
                resources.energy_remaining_j[uav_id] = max(
                    resources.energy_remaining_j[uav_id] - energy, 0.0
                )
                system_energy.cloud_reception_j += (
                    self.config.system_energy.cloud_receive_power_w
                    * actual_seconds
                )
            for portion in portions:
                record = self.tasks[portion.task_id]
                self.queues[cloud_staging_key].push(
                    WorkItem(
                        task_id=portion.task_id,
                        amount=portion.amount,
                        unit=QueueUnit.BITS,
                        intensity=record.spec.intensity,
                        branch_id="cloud",
                    )
                )
                departures[source_key.label] += portion.equivalent_cycles
                arrivals[cloud_staging_key.label] += portion.equivalent_cycles

    def _promote_received_tasks(
        self,
        departures: dict[str, float],
        arrivals: dict[str, float],
    ) -> None:
        pairs: list[tuple[QueueKey, QueueKey, QueueKey]] = []
        pairs.extend(
            (
                QueueKey(QueueStage.UPLOAD_RSU, node_id),
                QueueKey(QueueStage.RECEIVE_RSU, node_id),
                QueueKey(QueueStage.COMPUTE_RSU, node_id),
            )
            for node_id in range(len(self.config.rsu_cpu_hz))
        )
        pairs.extend(
            (
                QueueKey(QueueStage.UPLOAD_UAV, node_id),
                QueueKey(QueueStage.RECEIVE_UAV, node_id),
                QueueKey(QueueStage.COMPUTE_UAV, node_id),
            )
            for node_id in range(len(self.config.uav_cpu_hz))
        )
        for upload_key, staging_key, compute_key in pairs:
            upload = self.queues[upload_key]
            staging = self.queues[staging_key]
            for task_id in list(staging.task_ids()):
                if task_id in upload:
                    continue
                record = self.tasks[task_id]
                received_bits = staging.amount_for(task_id)
                if not isclose(
                    received_bits,
                    record.spec.input_bits,
                    rel_tol=self.config.conservation_rtol,
                    abs_tol=self.config.conservation_atol,
                ):
                    continue
                staged_item = staging.pop(task_id)
                departures[staging_key.label] += staged_item.equivalent_cycles
                if compute_key.stage is QueueStage.COMPUTE_UAV:
                    fraction = record.uav_fraction
                    if fraction is None:
                        raise AssertionError("UAV task is missing its fixed split")
                    uav_cycles = fraction * record.spec.cpu_cycles
                    cloud_bits = (1.0 - fraction) * record.spec.input_bits
                    if uav_cycles > 0:
                        self.queues[compute_key].push(
                            WorkItem(
                                task_id=task_id,
                                amount=uav_cycles,
                                unit=QueueUnit.CYCLES,
                                intensity=record.spec.intensity,
                                branch_id="uav",
                            )
                        )
                        arrivals[compute_key.label] += uav_cycles
                    if cloud_bits > 0:
                        backhaul_key = QueueKey(
                            QueueStage.BACKHAUL_UAV, compute_key.node_id
                        )
                        self.queues[backhaul_key].push(
                            WorkItem(
                                task_id=task_id,
                                amount=cloud_bits,
                                unit=QueueUnit.BITS,
                                intensity=record.spec.intensity,
                                branch_id="cloud",
                            )
                        )
                        arrivals[backhaul_key.label] += (
                            cloud_bits * record.spec.intensity
                        )
                else:
                    self.queues[compute_key].push(
                        WorkItem(
                            task_id=task_id,
                            amount=record.spec.cpu_cycles,
                            unit=QueueUnit.CYCLES,
                            intensity=record.spec.intensity,
                            branch_id="main",
                        )
                    )
                    arrivals[compute_key.label] += staged_item.equivalent_cycles
                record.state = TaskState.PROCESSING
                record.record_stage(self.slot + 1, compute_key.label)

    def _promote_cloud_received(
        self,
        departures: dict[str, float],
        arrivals: dict[str, float],
    ) -> None:
        staging_key = QueueKey(QueueStage.RECEIVE_CLOUD, 0)
        compute_key = QueueKey(QueueStage.COMPUTE_CLOUD, 0)
        staging = self.queues[staging_key]
        for task_id in list(staging.task_ids()):
            record = self.tasks[task_id]
            fraction = record.uav_fraction
            if fraction is None:
                raise AssertionError("cloud branch is missing its fixed split")
            backhaul_key = QueueKey(
                QueueStage.BACKHAUL_UAV, record.destination_id or 0
            )
            if task_id in self.queues[backhaul_key]:
                continue
            expected_bits = (1.0 - fraction) * record.spec.input_bits
            if not isclose(
                staging.amount_for(task_id),
                expected_bits,
                rel_tol=self.config.conservation_rtol,
                abs_tol=self.config.conservation_atol,
            ):
                continue
            item = staging.pop(task_id)
            cloud_cycles = (1.0 - fraction) * record.spec.cpu_cycles
            self.queues[compute_key].push(
                WorkItem(
                    task_id=task_id,
                    amount=cloud_cycles,
                    unit=QueueUnit.CYCLES,
                    intensity=record.spec.intensity,
                    branch_id="cloud",
                )
            )
            departures[staging_key.label] += item.equivalent_cycles
            arrivals[compute_key.label] += cloud_cycles
            record.record_stage(self.slot + 1, compute_key.label)

    def _commit_branch_completions(
        self,
        branches: Sequence[tuple[int, str]],
        end_time: float,
        boundary_slot: int,
    ) -> list[int]:
        completed: list[int] = []
        for task_id, branch_id in branches:
            record = self.tasks[task_id]
            record.complete_branch(branch_id)
            if record.all_branches_complete and record.merge_ready_time is None:
                record.merge_ready_time = end_time + self.config.merge_delay_s
                if self.config.merge_delay_s == 0:
                    self._finish_task(record, record.merge_ready_time, boundary_slot)
                    completed.append(task_id)
        return completed

    def _complete_ready_merges(
        self, end_time: float, boundary_slot: int
    ) -> list[int]:
        completed: list[int] = []
        for task_id in sorted(self.tasks):
            record = self.tasks[task_id]
            if record.is_complete or record.merge_ready_time is None:
                continue
            if record.merge_ready_time <= end_time + 1e-12:
                self._finish_task(record, record.merge_ready_time, boundary_slot)
                completed.append(task_id)
        return completed

    @staticmethod
    def _finish_task(
        record: TaskRecord, completion_time: float, boundary_slot: int
    ) -> None:
        record.state = TaskState.COMPLETED
        record.completion_time = completion_time
        record.record_stage(boundary_slot, TaskState.COMPLETED.value)

    def _record_deadline_crossings(self, time_now: float) -> list[int]:
        missed: list[int] = []
        for task_id in sorted(self.tasks):
            record = self.tasks[task_id]
            if record.is_complete or record.deadline_missed:
                continue
            if time_now - record.spec.arrival_time > record.spec.deadline:
                record.deadline_missed = True
                missed.append(task_id)
        return missed

    def _commit_scheduled_arrivals(self, slot: int) -> list[int]:
        committed: list[int] = []
        for arrival in self._arrival_trace.get(slot, ()):  # immutable iteration
            committed.append(
                self.submit_task(
                    vehicle_id=arrival.vehicle_id,
                    input_bits=arrival.input_bits,
                    cpu_cycles=arrival.cpu_cycles,
                    deadline=arrival.deadline,
                    task_id=arrival.task_id,
                    partitionable=arrival.partitionable,
                )
            )
        return committed

    def _allocate_task_id(self, requested: int | None) -> int:
        if requested is None:
            while self._next_task_id in self.tasks:
                self._next_task_id += 1
            task_id = self._next_task_id
        else:
            task_id = int(requested)
            if task_id < 0:
                raise ValueError("task_id must be non-negative")
            if task_id in self.tasks:
                raise ValueError(f"duplicate task_id {task_id}")
        self._next_task_id = max(self._next_task_id, task_id + 1)
        return task_id

    def _task(self, task_id: int) -> TaskRecord:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task_id {task_id}") from exc

    def _validate_vehicle(self, vehicle_id: int) -> None:
        if not 0 <= vehicle_id < len(self.config.vehicle_cpu_hz):
            raise ValueError(f"invalid vehicle_id {vehicle_id}")

    def _validate_destination(
        self, destination: Destination, vehicle_id: int
    ) -> None:
        if destination.node_id < 0:
            raise ValueError("destination node_id must be non-negative")
        if destination.kind is DestinationKind.LOCAL:
            if destination.node_id != vehicle_id:
                raise ValueError("local destination must be the source vehicle")
        elif destination.kind is DestinationKind.RSU:
            if destination.node_id >= len(self.config.rsu_cpu_hz):
                raise ValueError("invalid RSU destination")
            if self.config.radio is not None and not self.radio_scheduler.channel.in_rsu_coverage(
                self.vehicle_positions[vehicle_id],
                self.rsu_positions[destination.node_id],
            ):
                raise ValueError("RSU destination is outside current coverage")
        elif destination.kind is DestinationKind.UAV:
            if destination.node_id >= len(self.config.uav_cpu_hz):
                raise ValueError("invalid UAV destination")
            if not self._uav_service_enabled[destination.node_id]:
                raise ValueError("UAV destination is unavailable this slot")
            if self.config.radio is not None and not self.radio_scheduler.channel.in_uav_coverage(
                self.vehicle_positions[vehicle_id],
                self.uav_positions[destination.node_id],
            ):
                raise ValueError("UAV destination is outside current coverage")
        else:  # pragma: no cover - protects future enum extensions
            raise ValueError(f"unsupported destination {destination.kind}")

    @staticmethod
    def _dispatch_target(
        destination: Destination,
    ) -> tuple[QueueKey, QueueUnit]:
        if destination.kind is DestinationKind.LOCAL:
            return (
                QueueKey(QueueStage.COMPUTE_LOCAL, destination.node_id),
                QueueUnit.CYCLES,
            )
        if destination.kind is DestinationKind.RSU:
            return (
                QueueKey(QueueStage.UPLOAD_RSU, destination.node_id),
                QueueUnit.BITS,
            )
        return (
            QueueKey(QueueStage.UPLOAD_UAV, destination.node_id),
            QueueUnit.BITS,
        )

    @staticmethod
    def _node_positions(
        configured: Sequence[Position3D], count: int, *, altitude: float
    ) -> tuple[Position3D, ...]:
        if configured:
            return tuple(
                tuple(float(value) for value in position)  # type: ignore[misc]
                for position in configured
            )
        return tuple((0.0, 0.0, altitude) for _ in range(count))
