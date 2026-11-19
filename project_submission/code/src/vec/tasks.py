"""Persistent task records for the queue-aware VEC environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DestinationKind(str, Enum):
    """Execution destinations supported by the Phase-1 simulator."""

    LOCAL = "local"
    RSU = "rsu"
    UAV = "uav"


class TaskState(str, Enum):
    """Coarse task lifecycle state.

    Detailed workload placement is always available from the physical queues.
    A partially uploaded task can, for example, have work in both an upload
    queue and a receive/staging queue while its coarse state remains UPLOAD.
    """

    INGRESS = "ingress"
    UPLOAD = "upload"
    PROCESSING = "processing"
    COMPLETED = "completed"


@dataclass(frozen=True)
class TaskSpec:
    """Immutable task attributes.

    Times are seconds, input is bits, and work is CPU cycles.  ``deadline`` is
    a response-time allowance relative to ``arrival_time``.
    """

    task_id: int
    vehicle_id: int
    input_bits: float
    cpu_cycles: float
    arrival_time: float
    deadline: float
    partitionable: bool = False

    def __post_init__(self) -> None:
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.vehicle_id < 0:
            raise ValueError("vehicle_id must be non-negative")
        if self.input_bits <= 0:
            raise ValueError("input_bits must be positive")
        if self.cpu_cycles <= 0:
            raise ValueError("cpu_cycles must be positive")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.deadline <= 0:
            raise ValueError("deadline must be positive")

    @property
    def intensity(self) -> float:
        """Computation intensity in cycles per input bit."""

        return self.cpu_cycles / self.input_bits


@dataclass
class TaskRecord:
    """Mutable lifecycle information for one persistent task."""

    spec: TaskSpec
    state: TaskState = TaskState.INGRESS
    destination_kind: Optional[DestinationKind] = None
    destination_id: Optional[int] = None
    dispatch_time: Optional[float] = None
    completion_time: Optional[float] = None
    deadline_missed: bool = False
    uav_fraction: Optional[float] = None
    required_branches: tuple[str, ...] = ()
    completed_branches: set[str] = field(default_factory=set)
    merge_ready_time: Optional[float] = None
    stage_history: list[tuple[int, str]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.state is TaskState.COMPLETED

    @property
    def latency(self) -> Optional[float]:
        if self.completion_time is None:
            return None
        return self.completion_time - self.spec.arrival_time

    def record_stage(self, slot: int, stage: str) -> None:
        entry = (int(slot), str(stage))
        if not self.stage_history or self.stage_history[-1] != entry:
            self.stage_history.append(entry)

    def configure_branches(self, uav_fraction: float | None = None) -> None:
        if uav_fraction is None:
            self.uav_fraction = None
            self.required_branches = ("main",)
            return
        if not 0.0 <= uav_fraction <= 1.0:
            raise ValueError("uav_fraction must lie in [0, 1]")
        self.uav_fraction = uav_fraction
        branches: list[str] = []
        if uav_fraction > 0:
            branches.append("uav")
        if uav_fraction < 1:
            branches.append("cloud")
        self.required_branches = tuple(branches)

    def complete_branch(self, branch_id: str) -> None:
        if branch_id not in self.required_branches:
            raise ValueError(f"unexpected branch {branch_id!r}")
        self.completed_branches.add(branch_id)

    @property
    def all_branches_complete(self) -> bool:
        return bool(self.required_branches) and all(
            branch in self.completed_branches for branch in self.required_branches
        )
