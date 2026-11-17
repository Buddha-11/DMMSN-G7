"""Reptile meta-learning and bounded change-point adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pickle import dumps
from typing import Callable, Iterable

import torch

from .agent import HybridMADDPG, UpdateDiagnostics
from .replay import HybridTransition


@dataclass(frozen=True)
class AdaptationResult:
    learner: HybridMADDPG
    diagnostics: tuple[UpdateDiagnostics, ...]


class ReptileMetaLearner:
    """First-order Reptile over all online actor and critic parameters."""

    def __init__(self, learner: HybridMADDPG, outer_step_size: float = 0.1) -> None:
        if not 0.0 < outer_step_size <= 1.0:
            raise ValueError("outer_step_size must be in (0, 1]")
        self.learner = learner
        self.outer_step_size = outer_step_size

    def adapt_task(
        self,
        task_id: str,
        transitions: Iterable[HybridTransition],
        inner_updates: int,
    ) -> AdaptationResult:
        if inner_updates < 0:
            raise ValueError("inner_updates cannot be negative")
        adapted = self.learner.fork()
        adapted.begin_meta_task(task_id)
        for transition in transitions:
            adapted.remember(transition)
        diagnostics: list[UpdateDiagnostics] = []
        for _ in range(inner_updates):
            if not adapted.ready():
                raise ValueError("task has insufficient replay for inner updates")
            diagnostics.append(adapted.update())
        return AdaptationResult(adapted, tuple(diagnostics))

    def outer_update(self, adapted_learners: Iterable[HybridMADDPG]) -> None:
        adapted = tuple(adapted_learners)
        if not adapted:
            raise ValueError("at least one adapted learner is required")
        for candidate in adapted:
            if candidate.config != self.learner.config:
                raise ValueError("all Reptile learners must share one configuration")
        rate = self.outer_step_size
        with torch.no_grad():
            for meta_module_name in ("actors", "critics"):
                meta_module = getattr(self.learner, meta_module_name)
                candidate_parameters = [
                    dict(getattr(item, meta_module_name).named_parameters())
                    for item in adapted
                ]
                for name, parameter in meta_module.named_parameters():
                    mean = torch.stack(
                        [values[name].detach() for values in candidate_parameters]
                    ).mean(dim=0)
                    parameter.add_(mean - parameter, alpha=rate)
        # Inner-loop moments must never leak between scenarios. Targets are
        # hard-synchronized at the outer boundary as required by Reptile.
        self.learner.reset_optimizers()
        self.learner.synchronize_targets()


def vec_physical_state_token(environment: object) -> str:
    """Hash the physical VEC state that online adaptation must not reset.

    The learner is intentionally absent from this snapshot. This catches
    accidental calls to ``env.reset()`` as well as queue, clock, vehicle/UAV,
    battery, and task-record mutations during parameter-only adaptation.
    """

    attributes = (
        "slot",
        "vehicle_positions",
        "uav_positions",
        "uav_states",
        "batteries",
        "queues",
        "tasks",
        "uav_modes",
        "lyapunov_tracker",
    )
    snapshot = tuple(
        (name, getattr(environment, name))
        for name in attributes
        if hasattr(environment, name)
    )
    if not snapshot:
        raise TypeError("environment exposes no recognized physical VEC state")
    return sha256(dumps(snapshot, protocol=5)).hexdigest()


@dataclass(frozen=True)
class OnlineAdaptationReport:
    change_id: str
    requested_updates: int
    completed_updates: int
    remaining_budget: int
    diagnostics: tuple[UpdateDiagnostics, ...]


class BoundedOnlineAdapter:
    """Limit post-change gradient work without advancing/resetting the env."""

    def __init__(
        self,
        learner: HybridMADDPG,
        max_updates_per_change: int,
        state_token: Callable[[object], str] = vec_physical_state_token,
    ) -> None:
        if max_updates_per_change < 0:
            raise ValueError("max_updates_per_change cannot be negative")
        self.learner = learner
        self.max_updates_per_change = max_updates_per_change
        self.state_token = state_token
        self._change_id: str | None = None
        self._environment: object | None = None
        self._initial_token: str | None = None
        self._updates_used = 0

    def declare_change(self, change_id: str, environment: object) -> None:
        if not change_id:
            raise ValueError("change_id must be non-empty")
        self._change_id = change_id
        self._environment = environment
        self._initial_token = self.state_token(environment)
        self._updates_used = 0

    def adapt(self, requested_updates: int) -> OnlineAdaptationReport:
        if self._change_id is None or self._environment is None:
            raise RuntimeError("declare a change point before online adaptation")
        if requested_updates < 0:
            raise ValueError("requested_updates cannot be negative")
        before = self.state_token(self._environment)
        if before != self._initial_token:
            raise RuntimeError("environment advanced after the change declaration")
        allowed = min(
            requested_updates,
            self.max_updates_per_change - self._updates_used,
        )
        diagnostics: list[UpdateDiagnostics] = []
        for _ in range(allowed):
            if not self.learner.ready():
                break
            diagnostics.append(self.learner.update())
        self._updates_used += len(diagnostics)
        after = self.state_token(self._environment)
        if after != before:
            raise RuntimeError(
                "online adaptation changed physical environment state"
            )
        return OnlineAdaptationReport(
            change_id=self._change_id,
            requested_updates=requested_updates,
            completed_updates=len(diagnostics),
            remaining_budget=self.max_updates_per_change - self._updates_used,
            diagnostics=tuple(diagnostics),
        )
