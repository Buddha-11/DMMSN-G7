"""Work-conserving queue primitives with explicit physical units."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping


_EPS = 1e-9


class QueueUnit(str, Enum):
    BITS = "bits"
    CYCLES = "cycles"


class QueueStage(str, Enum):
    INGRESS = "ingress"
    UPLOAD_RSU = "upload_rsu"
    UPLOAD_UAV = "upload_uav"
    RECEIVE_RSU = "receive_rsu"
    RECEIVE_UAV = "receive_uav"
    COMPUTE_LOCAL = "compute_local"
    COMPUTE_RSU = "compute_rsu"
    COMPUTE_UAV = "compute_uav"
    BACKHAUL_UAV = "backhaul_uav"
    RECEIVE_CLOUD = "receive_cloud"
    COMPUTE_CLOUD = "compute_cloud"


@dataclass(frozen=True, order=True)
class QueueKey:
    stage: QueueStage
    node_id: int

    @property
    def label(self) -> str:
        return f"{self.stage.value}:{self.node_id}"


@dataclass
class WorkItem:
    """A task's residual work in a single queue."""

    task_id: int
    amount: float
    unit: QueueUnit
    intensity: float
    branch_id: str = "main"

    def __post_init__(self) -> None:
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if not isfinite(self.amount) or self.amount < 0:
            raise ValueError("amount must be finite and non-negative")
        if not isfinite(self.intensity) or self.intensity <= 0:
            raise ValueError("intensity must be finite and positive")
        if not self.branch_id:
            raise ValueError("branch_id must be non-empty")

    @property
    def equivalent_cycles(self) -> float:
        if self.unit is QueueUnit.CYCLES:
            return self.amount
        return self.amount * self.intensity


@dataclass(frozen=True)
class ServicePortion:
    task_id: int
    amount: float
    equivalent_cycles: float
    completed_item: bool
    branch_id: str


class WorkQueue:
    """Deterministic FIFO storage with FIFO and weighted service policies."""

    def __init__(self, key: QueueKey, unit: QueueUnit):
        self.key = key
        self.unit = unit
        self._items: "OrderedDict[int, WorkItem]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, task_id: int) -> bool:
        return task_id in self._items

    def task_ids(self) -> tuple[int, ...]:
        return tuple(self._items)

    def get(self, task_id: int) -> WorkItem | None:
        return self._items.get(task_id)

    def amount_for(self, task_id: int) -> float:
        item = self._items.get(task_id)
        return 0.0 if item is None else item.amount

    @property
    def native_backlog(self) -> float:
        return sum(item.amount for item in self._items.values())

    @property
    def equivalent_backlog(self) -> float:
        return sum(item.equivalent_cycles for item in self._items.values())

    def push(self, item: WorkItem) -> None:
        if item.unit is not self.unit:
            raise ValueError(
                f"queue {self.key.label} stores {self.unit.value}, "
                f"not {item.unit.value}"
            )
        if item.amount <= _EPS:
            return
        existing = self._items.get(item.task_id)
        if existing is None:
            self._items[item.task_id] = item
            return
        if abs(existing.intensity - item.intensity) > _EPS:
            raise ValueError("cannot merge work with different intensities")
        if existing.branch_id != item.branch_id:
            raise ValueError("cannot merge work from different branches")
        existing.amount += item.amount

    def pop(self, task_id: int) -> WorkItem:
        try:
            return self._items.pop(task_id)
        except KeyError as exc:
            raise KeyError(
                f"task {task_id} is not in queue {self.key.label}"
            ) from exc

    def serve_fifo(self, capacity: float) -> list[ServicePortion]:
        """Serve up to ``capacity`` native units in insertion order."""

        self._validate_capacity(capacity)
        remaining_capacity = capacity
        portions: list[ServicePortion] = []
        for task_id in list(self._items):
            if remaining_capacity <= _EPS:
                break
            item = self._items[task_id]
            served = min(item.amount, remaining_capacity)
            item.amount -= served
            remaining_capacity -= served
            completed = item.amount <= _EPS
            portions.append(
                ServicePortion(
                    task_id=task_id,
                    amount=served,
                    equivalent_cycles=self._to_cycles(served, item.intensity),
                    completed_item=completed,
                    branch_id=item.branch_id,
                )
            )
            if completed:
                del self._items[task_id]
        return portions

    def serve_task(self, task_id: int, capacity: float) -> ServicePortion | None:
        """Serve one explicitly scheduled item without changing queue order."""

        self._validate_capacity(capacity)
        item = self._items.get(task_id)
        if item is None or capacity <= _EPS:
            return None
        served = min(item.amount, capacity)
        item.amount -= served
        completed = item.amount <= _EPS
        portion = ServicePortion(
            task_id=task_id,
            amount=served,
            equivalent_cycles=self._to_cycles(served, item.intensity),
            completed_item=completed,
            branch_id=item.branch_id,
        )
        if completed:
            del self._items[task_id]
        return portion

    def serve_weighted(
        self,
        capacity: float,
        weights: Mapping[int, float] | None = None,
    ) -> list[ServicePortion]:
        """Serve a cycle queue using work-conserving weighted sharing.

        Unused capacity from tasks that finish early is redistributed among the
        remaining positive-weight tasks.  Equal weights are used by default.
        """

        if self.unit is not QueueUnit.CYCLES:
            raise ValueError("weighted service is defined for cycle queues")
        self._validate_capacity(capacity)
        if capacity <= _EPS or not self._items:
            return []

        supplied = weights or {}
        task_weights: dict[int, float] = {}
        for task_id in self._items:
            weight = float(supplied.get(task_id, 1.0))
            if not isfinite(weight) or weight < 0:
                raise ValueError("CPU weights must be finite and non-negative")
            task_weights[task_id] = weight

        served_by_task = {task_id: 0.0 for task_id in self._items}
        branches = {
            task_id: item.branch_id for task_id, item in self._items.items()
        }
        active = [task_id for task_id in self._items if task_weights[task_id] > 0]
        remaining_capacity = capacity

        while active and remaining_capacity > _EPS:
            total_weight = sum(task_weights[task_id] for task_id in active)
            if total_weight <= _EPS:
                break
            round_capacity = remaining_capacity
            finished_any = False
            for task_id in list(active):
                item = self._items[task_id]
                fair_share = round_capacity * task_weights[task_id] / total_weight
                served = min(item.amount, fair_share)
                item.amount -= served
                remaining_capacity -= served
                served_by_task[task_id] += served
                if item.amount <= _EPS:
                    del self._items[task_id]
                    active.remove(task_id)
                    finished_any = True
            if not finished_any:
                break

        return [
            ServicePortion(
                task_id=task_id,
                amount=amount,
                equivalent_cycles=amount,
                completed_item=task_id not in self._items,
                branch_id=branches[task_id],
            )
            for task_id, amount in served_by_task.items()
            if amount > _EPS
        ]

    def _to_cycles(self, amount: float, intensity: float) -> float:
        if self.unit is QueueUnit.CYCLES:
            return amount
        return amount * intensity

    @staticmethod
    def _validate_capacity(capacity: float) -> None:
        if not isfinite(capacity) or capacity < 0:
            raise ValueError("capacity must be finite and non-negative")
