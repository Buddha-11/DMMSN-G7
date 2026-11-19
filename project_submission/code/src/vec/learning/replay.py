"""Scenario-isolated replay for hybrid multi-agent transitions."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class HybridTransition:
    state: np.ndarray
    next_state: np.ndarray
    observations: Mapping[str, np.ndarray]
    next_observations: Mapping[str, np.ndarray]
    task_features: Mapping[str, np.ndarray]
    next_task_features: Mapping[str, np.ndarray]
    task_masks: Mapping[str, np.ndarray]
    next_task_masks: Mapping[str, np.ndarray]
    categorical_masks: Mapping[str, np.ndarray]
    next_categorical_masks: Mapping[str, np.ndarray]
    proposed_actions: Mapping[str, np.ndarray]
    executed_actions: Mapping[str, np.ndarray]
    rewards: Mapping[str, float]
    done: bool


class TaskLocalReplayBank:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.seed = seed
        self._buffers: dict[str, deque[HybridTransition]] = {}
        self._random = Random(seed)

    def reset_task(self, task_id: str) -> None:
        self._buffers[task_id] = deque(maxlen=self.capacity)

    def clear(self) -> None:
        self._buffers.clear()

    def push(self, task_id: str, transition: HybridTransition) -> None:
        if task_id not in self._buffers:
            self.reset_task(task_id)
        self._buffers[task_id].append(deepcopy(transition))

    def sample(self, task_id: str, batch_size: int) -> list[HybridTransition]:
        buffer = self._buffers.get(task_id)
        if buffer is None or len(buffer) < batch_size:
            raise ValueError("not enough task-local replay samples")
        indices = self._random.sample(range(len(buffer)), batch_size)
        return [deepcopy(buffer[index]) for index in indices]

    def size(self, task_id: str) -> int:
        return len(self._buffers.get(task_id, ()))

    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._buffers))

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "buffers": {
                task_id: list(buffer) for task_id, buffer in self._buffers.items()
            },
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("replay capacity mismatch")
        self._buffers = {
            task_id: deque(items, maxlen=self.capacity)
            for task_id, items in state["buffers"].items()
        }
        self._random.setstate(state["random_state"])
