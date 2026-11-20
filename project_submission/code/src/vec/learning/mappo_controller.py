"""Common evaluation-controller wrapper for hybrid MAPPO checkpoints."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from ..baselines import BaselineDecision, BaselineDiagnostics
from ..env import StepResult, VecQueueEnv
from .adapter import VecPolicyAdapter
from .mappo import HybridMAPPO


class HybridMAPPOController:
    def __init__(
        self,
        name: str,
        learner: HybridMAPPO,
        environment: VecQueueEnv,
        *,
        deterministic: bool = True,
    ) -> None:
        self.name = name
        self.learner = learner
        self.environment = environment
        self.deterministic = deterministic
        self.adapter = VecPolicyAdapter(
            environment, max_tasks=max(spec.max_tasks for spec in learner.role_specs)
        )
        if self.adapter.role_specs() != learner.role_specs:
            raise ValueError("checkpoint role specification does not match environment")

    @classmethod
    def from_checkpoint(
        cls,
        name: str,
        checkpoint_path: str | Path,
        environment: VecQueueEnv,
        *,
        deterministic: bool = True,
    ) -> "HybridMAPPOController":
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        learner = HybridMAPPO(checkpoint["config"])
        learner.load(checkpoint_path)
        return cls(name, learner, environment, deterministic=deterministic)

    def begin_scenario(self, scenario_id: str) -> None:
        del scenario_id

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        if env is not self.environment:
            raise ValueError("controller is bound to a different environment")
        started = perf_counter()
        encoded = self.adapter.encode()
        actions = self.learner.policy_actions(
            encoded.observations,
            encoded.task_features,
            encoded.task_masks,
            encoded.categorical_masks,
            deterministic=self.deterministic,
        )
        return BaselineDecision(
            action=self.adapter.decode(encoded, actions),
            diagnostics=BaselineDiagnostics(
                controller=self.name,
                candidates_evaluated=1,
                evaluation_budget=None,
                elapsed_seconds=perf_counter() - started,
                seed=self.learner.config.seed,
            ),
        )

    def observe_transition(self, result: StepResult) -> None:
        del result
