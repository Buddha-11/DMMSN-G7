"""Controller-neutral observations, masks, and executed-action records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .safety import Velocity3D


@dataclass(frozen=True)
class AgentObservation:
    role: str
    agent_id: int
    features: Mapping[str, float]
    eligible_task_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DestinationMask:
    vehicle_id: int
    hold: bool
    local: bool
    rsu: tuple[bool, ...]
    uav: tuple[bool, ...]
    ingress_task_ids: tuple[int, ...]


@dataclass(frozen=True)
class ControlObservation:
    vehicles: tuple[AgentObservation, ...]
    rsus: tuple[AgentObservation, ...]
    uavs: tuple[AgentObservation, ...]
    global_critic_state: tuple[float, ...]
    global_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class DispatchTrace:
    vehicle_id: int
    task_id: int
    destination_kind: str
    destination_id: int
    uav_fraction: float


@dataclass(frozen=True)
class ExecutedActionTrace:
    dispatches: tuple[DispatchTrace, ...]
    proposed_uav_velocities: tuple[Velocity3D, ...]
    executed_uav_velocities: tuple[Velocity3D, ...]
    requested_uav_cpu_hz: tuple[float, ...]
    executed_uav_cpu_hz: tuple[float, ...]
    motion_correction_mps: tuple[float, ...]
    motion_candidates_checked: int
    physical_feasibility_failure: bool
