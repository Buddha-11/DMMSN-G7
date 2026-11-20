"""Versioned experiment configuration for VEC policy training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Literal


TRAINING_SCHEMA_VERSION = "vec-training/v1"
TrainingMode = Literal["maddpg", "domain_randomized", "reptile"]


@dataclass(frozen=True)
class TrainingConfig:
    mode: TrainingMode = "domain_randomized"
    seed: int = 0
    epochs: int = 50
    scenarios_per_epoch: int = 7
    base_stress: str = "near_capacity"
    train_stresses: tuple[str, ...] = (
        "low_load",
        "near_capacity",
        "overload",
        "zero_harvest",
        "harvest_outage",
        "backhaul_bottleneck",
        "traffic_surge",
    )
    max_tasks: int = 16
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    actor_lr: float = 1.0e-3
    critic_lr: float = 1.0e-3
    gamma: float = 0.99
    tau: float = 0.01
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 64
    updates_per_step: int = 1
    exploration_std: float = 0.15
    exploration_decay: float = 0.995
    validation_every: int = 1
    validation_scenarios: int = 35
    early_stopping_patience: int = 10
    reptile_inner_updates: int = 8
    reptile_outer_step_size: float = 0.1
    no_queue_weight: bool = False
    device: str = "cpu"
    schema_version: str = TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_SCHEMA_VERSION:
            raise ValueError("unsupported training configuration schema")
        if self.mode not in {"maddpg", "domain_randomized", "reptile"}:
            raise ValueError(f"unsupported training mode {self.mode!r}")
        positive = (
            self.epochs,
            self.scenarios_per_epoch,
            self.max_tasks,
            self.actor_hidden_dim,
            self.critic_hidden_dim,
            self.batch_size,
            self.replay_capacity,
            self.validation_every,
            self.validation_scenarios,
            self.early_stopping_patience,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training dimensions and intervals must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.replay_capacity < self.warmup_steps:
            raise ValueError("replay_capacity must be at least warmup_steps")
        if self.warmup_steps < 0 or self.updates_per_step < 0:
            raise ValueError("warmup and update counts cannot be negative")
        if self.reptile_inner_updates < 0:
            raise ValueError("Reptile inner updates cannot be negative")
        if (
            not isfinite(self.exploration_std)
            or self.exploration_std < 0.0
            or not isfinite(self.exploration_decay)
            or not 0.0 < self.exploration_decay <= 1.0
        ):
            raise ValueError("invalid exploration schedule")
        if (
            not isfinite(self.reptile_outer_step_size)
            or not 0.0 < self.reptile_outer_step_size <= 1.0
        ):
            raise ValueError("Reptile outer step size must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma or target-update rate is invalid")
        if (
            not isfinite(self.actor_lr)
            or not isfinite(self.critic_lr)
            or self.actor_lr <= 0.0
            or self.critic_lr <= 0.0
        ):
            raise ValueError("learning rates must be finite and positive")
        if not self.base_stress or not self.train_stresses:
            raise ValueError("at least one named training stress is required")
        object.__setattr__(self, "train_stresses", tuple(self.train_stresses))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["train_stresses"] = tuple(payload["train_stresses"])
        return cls(**payload)
