"""Common-controller wrapper for evaluating hybrid MADDPG checkpoints."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from ..baselines import BaselineDecision, BaselineDiagnostics
from ..env import StepResult, VecQueueEnv
from .adapter import EncodedControlObservation, VecPolicyAdapter
from .agent import HybridMADDPG
from .meta import BoundedOnlineAdapter, OnlineAdaptationReport


class HybridPolicyController:
    """Run a learned policy through the same StepAction path as baselines."""

    def __init__(
        self,
        name: str,
        learner: HybridMADDPG,
        environment: VecQueueEnv,
        *,
        deterministic: bool = True,
        max_online_updates: int = 0,
    ) -> None:
        if not name:
            raise ValueError("controller name cannot be empty")
        self.name = name
        self.learner = learner
        self.environment = environment
        max_tasks = max(spec.max_tasks for spec in learner.role_specs)
        self.adapter = VecPolicyAdapter(environment, max_tasks=max_tasks)
        if self.adapter.role_specs() != learner.role_specs:
            raise ValueError("checkpoint role specification does not match environment")
        if len(environment.control_observation().global_critic_state) != (
            learner.config.state_dim
        ):
            raise ValueError("checkpoint critic state dimension does not match environment")
        self.deterministic = deterministic
        self._pending: tuple[EncodedControlObservation, dict] | None = None
        self._online = BoundedOnlineAdapter(learner, max_online_updates)

    @classmethod
    def from_checkpoint(
        cls,
        name: str,
        checkpoint_path: str | Path,
        environment: VecQueueEnv,
        *,
        deterministic: bool = True,
        max_online_updates: int = 0,
    ) -> "HybridPolicyController":
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        learner = HybridMADDPG(checkpoint["config"])
        learner.load(checkpoint_path)
        return cls(
            name,
            learner,
            environment,
            deterministic=deterministic,
            max_online_updates=max_online_updates,
        )

    def begin_scenario(self, scenario_id: str) -> None:
        self.learner.begin_meta_task(scenario_id)
        self._pending = None

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        if env is not self.environment:
            raise ValueError("controller is bound to a different environment")
        start = perf_counter()
        encoded = self.adapter.encode()
        actions = self.learner.act(
            encoded.observations,
            encoded.task_features,
            encoded.task_masks,
            encoded.categorical_masks,
            deterministic=self.deterministic,
        )
        action = self.adapter.decode(encoded, actions)
        elapsed = perf_counter() - start
        self._pending = (encoded, actions)
        return BaselineDecision(
            action=action,
            diagnostics=BaselineDiagnostics(
                controller=self.name,
                candidates_evaluated=1,
                evaluation_budget=None,
                elapsed_seconds=elapsed,
                seed=self.learner.config.seed,
            ),
        )

    def observe_transition(self, result: StepResult) -> None:
        if self._pending is None:
            raise RuntimeError("select_action must precede observe_transition")
        before, proposed = self._pending
        after = self.adapter.encode(result.observation_after, result.masks_after)
        self.learner.remember(
            self.adapter.transition(before, after, proposed, result)
        )
        self._pending = None

    def declare_change(self, change_id: str, environment: VecQueueEnv) -> None:
        self._online.declare_change(change_id, environment)

    def adapt(self, requested_updates: int) -> OnlineAdaptationReport:
        return self._online.adapt(requested_updates)
