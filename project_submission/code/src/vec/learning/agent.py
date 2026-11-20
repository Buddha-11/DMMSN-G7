"""Hybrid role-shared MADDPG learner for the queue-aware VEC simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .networks import CentralizedCritic, HybridActionRepair, HybridRoleActor, RoleSpec
from .replay import HybridTransition, TaskLocalReplayBank


@dataclass(frozen=True)
class HybridMADDPGConfig:
    state_dim: int
    role_specs: tuple[RoleSpec, ...]
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1.0e-3
    critic_lr: float = 1.0e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    gumbel_temperature: float = 1.0
    repair_regularization: float = 0.05
    max_gradient_norm: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not self.role_specs:
            raise ValueError("at least one role is required")
        names = [spec.name for spec in self.role_specs]
        if len(names) != len(set(names)):
            raise ValueError("role names must be unique")
        if any(spec.count <= 0 or spec.action_dim <= 0 for spec in self.role_specs):
            raise ValueError("every role needs agents and at least one action head")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must be at least one positive batch")

    @property
    def joint_action_dim(self) -> int:
        return sum(spec.count * spec.action_dim for spec in self.role_specs)


@dataclass(frozen=True)
class UpdateDiagnostics:
    critic_loss: Mapping[str, float]
    actor_loss: Mapping[str, float]
    repair_loss: Mapping[str, float]
    critic_gradient_norm: Mapping[str, float]
    actor_gradient_norm: Mapping[str, float]


class HybridMADDPG:
    """Centralized-training/decentralized-execution hybrid MADDPG.

    A single actor is shared by all agents in each role. Each role owns a
    centralized critic, so heterogeneous role rewards can be optimized while
    every critic still sees the complete state and joint executed action.
    """

    def __init__(self, config: HybridMADDPGConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self._np_random = np.random.default_rng(config.seed)
        self._specs = {spec.name: spec for spec in config.role_specs}
        self.actors = nn.ModuleDict(
            {
                spec.name: HybridRoleActor(spec, config.actor_hidden_dim)
                for spec in config.role_specs
            }
        ).to(self.device)
        self.target_actors = deepcopy(self.actors).to(self.device)
        self.critics = nn.ModuleDict(
            {
                spec.name: CentralizedCritic(
                    config.state_dim,
                    config.joint_action_dim,
                    config.critic_hidden_dim,
                )
                for spec in config.role_specs
            }
        ).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)
        self._repair = HybridActionRepair()
        self.replay = TaskLocalReplayBank(config.replay_capacity, config.seed)
        self.current_task_id: str | None = None
        self.update_count = 0
        self._reset_optimizers()
        self.synchronize_targets()

    @property
    def role_specs(self) -> tuple[RoleSpec, ...]:
        return self.config.role_specs

    def _reset_optimizers(self) -> None:
        self.actor_optimizers = {
            name: torch.optim.Adam(actor.parameters(), lr=self.config.actor_lr)
            for name, actor in self.actors.items()
        }
        self.critic_optimizers = {
            name: torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr)
            for name, critic in self.critics.items()
        }

    def reset_optimizers(self) -> None:
        """Discard task-specific Adam moments while retaining model weights."""

        self._reset_optimizers()

    def begin_meta_task(self, task_id: str) -> None:
        """Start a clean inner loop for one scenario."""

        if not task_id:
            raise ValueError("task_id must be non-empty")
        self.current_task_id = task_id
        self.replay.clear()
        self.replay.reset_task(task_id)
        self.reset_optimizers()
        self.synchronize_targets()

    def remember(self, transition: HybridTransition) -> None:
        if self.current_task_id is None:
            raise RuntimeError("call begin_meta_task before adding replay")
        self._validate_transition(transition)
        self.replay.push(self.current_task_id, transition)

    def ready(self) -> bool:
        return (
            self.current_task_id is not None
            and self.replay.size(self.current_task_id) >= self.config.batch_size
        )

    def act(
        self,
        observations: Mapping[str, np.ndarray | torch.Tensor],
        task_features: Mapping[str, np.ndarray | torch.Tensor],
        task_masks: Mapping[str, np.ndarray | torch.Tensor],
        categorical_masks: Mapping[str, np.ndarray | torch.Tensor],
        *,
        deterministic: bool = False,
        use_targets: bool = False,
    ) -> dict[str, np.ndarray]:
        modules = self.target_actors if use_targets else self.actors
        result: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for spec in self.role_specs:
                observation = self._tensor(observations[spec.name]).unsqueeze(0)
                features = self._tensor(task_features[spec.name]).unsqueeze(0)
                task_mask = self._bool_tensor(task_masks[spec.name]).unsqueeze(0)
                categorical_mask = self._bool_tensor(
                    categorical_masks[spec.name]
                ).unsqueeze(0)
                action = modules[spec.name](
                    observation,
                    features,
                    task_mask,
                    categorical_mask,
                    temperature=self.config.gumbel_temperature,
                    deterministic=deterministic,
                )
                result[spec.name] = action.squeeze(0).cpu().numpy()
        return result

    def update(self) -> UpdateDiagnostics:
        if not self.ready() or self.current_task_id is None:
            raise RuntimeError("not enough task-local replay for an update")
        transitions = self.replay.sample(
            self.current_task_id, self.config.batch_size
        )
        batch = self._collate(transitions)

        with torch.no_grad():
            target_actions: dict[str, torch.Tensor] = {}
            for spec in self.role_specs:
                proposal = self.target_actors[spec.name](
                    batch["next_observations"][spec.name],
                    batch["next_task_features"][spec.name],
                    batch["next_task_masks"][spec.name],
                    batch["next_categorical_masks"][spec.name],
                    temperature=self.config.gumbel_temperature,
                    deterministic=True,
                )
                target_actions[spec.name] = self._repair.repair(
                    spec,
                    proposal,
                    batch["next_task_masks"][spec.name],
                    batch["next_categorical_masks"][spec.name],
                )
            target_joint = self._flatten_joint(target_actions)

        critic_losses: dict[str, float] = {}
        critic_norms: dict[str, float] = {}
        for spec in self.role_specs:
            optimizer = self.critic_optimizers[spec.name]
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                next_q = self.target_critics[spec.name](
                    batch["next_state"], target_joint
                )
                target = batch["rewards"][spec.name] + (
                    self.config.gamma * (1.0 - batch["done"]) * next_q
                )
            predicted = self.critics[spec.name](
                batch["state"], batch["executed_joint"]
            )
            loss = F.mse_loss(predicted, target)
            loss.backward()
            critic_norms[spec.name] = self._gradient_norm(
                self.critics[spec.name]
            )
            nn.utils.clip_grad_norm_(
                self.critics[spec.name].parameters(),
                self.config.max_gradient_norm,
            )
            optimizer.step()
            critic_losses[spec.name] = float(loss.detach().cpu())

        actor_losses: dict[str, float] = {}
        repair_losses: dict[str, float] = {}
        actor_norms: dict[str, float] = {}
        for spec in self.role_specs:
            critic = self.critics[spec.name]
            self._set_requires_grad(critic, False)
            optimizer = self.actor_optimizers[spec.name]
            optimizer.zero_grad(set_to_none=True)
            proposal = self.actors[spec.name](
                batch["observations"][spec.name],
                batch["task_features"][spec.name],
                batch["task_masks"][spec.name],
                batch["categorical_masks"][spec.name],
                temperature=self.config.gumbel_temperature,
                deterministic=False,
            )
            joint_parts = {
                other.name: (
                    proposal
                    if other.name == spec.name
                    else batch["executed_actions"][other.name].detach()
                )
                for other in self.role_specs
            }
            repair_loss = F.mse_loss(
                proposal, batch["executed_actions"][spec.name]
            )
            policy_loss = -critic(
                batch["state"], self._flatten_joint(joint_parts)
            ).mean()
            loss = policy_loss + self.config.repair_regularization * repair_loss
            loss.backward()
            actor_norms[spec.name] = self._gradient_norm(
                self.actors[spec.name]
            )
            nn.utils.clip_grad_norm_(
                self.actors[spec.name].parameters(),
                self.config.max_gradient_norm,
            )
            optimizer.step()
            self._set_requires_grad(critic, True)
            actor_losses[spec.name] = float(loss.detach().cpu())
            repair_losses[spec.name] = float(repair_loss.detach().cpu())

        self.polyak_update()
        self.update_count += 1
        return UpdateDiagnostics(
            critic_loss=critic_losses,
            actor_loss=actor_losses,
            repair_loss=repair_losses,
            critic_gradient_norm=critic_norms,
            actor_gradient_norm=actor_norms,
        )

    def synchronize_targets(self) -> None:
        self.target_actors.load_state_dict(self.actors.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

    def polyak_update(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for online, target in (
                (self.actors, self.target_actors),
                (self.critics, self.target_critics),
            ):
                for target_parameter, parameter in zip(
                    target.parameters(), online.parameters(), strict=True
                ):
                    target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def online_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "actors": deepcopy(self.actors.state_dict()),
            "critics": deepcopy(self.critics.state_dict()),
        }

    def load_online_state_dict(
        self, state: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> None:
        self.actors.load_state_dict(state["actors"])
        self.critics.load_state_dict(state["critics"])

    def fork(self) -> "HybridMADDPG":
        # Construction seeds module initialization. Preserve the caller's
        # global Torch stream so cloning cannot repeat exploration noise or
        # perturb training merely by making a policy copy.
        torch_rng_state = torch.get_rng_state()
        clone = HybridMADDPG(self.config)
        torch.set_rng_state(torch_rng_state)
        clone.load_online_state_dict(self.online_state_dict())
        clone.synchronize_targets()
        return clone

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "algorithm": "hybrid_maddpg",
            "config": self.config,
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizers": {
                key: value.state_dict()
                for key, value in self.actor_optimizers.items()
            },
            "critic_optimizers": {
                key: value.state_dict()
                for key, value in self.critic_optimizers.items()
            },
            "replay": self.replay.state_dict(),
            "current_task_id": self.current_task_id,
            "update_count": self.update_count,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": deepcopy(self._np_random.bit_generator.state),
        }
        torch.save(checkpoint, Path(path))

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        if checkpoint["config"] != self.config:
            raise ValueError("checkpoint configuration does not match learner")
        self.actors.load_state_dict(checkpoint["actors"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        for name, state in checkpoint["actor_optimizers"].items():
            self.actor_optimizers[name].load_state_dict(state)
        for name, state in checkpoint["critic_optimizers"].items():
            self.critic_optimizers[name].load_state_dict(state)
        self.replay.load_state_dict(checkpoint["replay"])
        self.current_task_id = checkpoint["current_task_id"]
        self.update_count = checkpoint["update_count"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        self._np_random.bit_generator.state = checkpoint["numpy_rng_state"]

    def _validate_transition(self, transition: HybridTransition) -> None:
        if np.asarray(transition.state).shape != (self.config.state_dim,):
            raise ValueError("transition state shape does not match state_dim")
        if np.asarray(transition.next_state).shape != (self.config.state_dim,):
            raise ValueError("transition next_state shape does not match state_dim")
        for spec in self.role_specs:
            name = spec.name
            expected = {
                "observations": (spec.count, spec.observation_dim),
                "next_observations": (spec.count, spec.observation_dim),
                "task_features": (
                    spec.count,
                    spec.max_tasks,
                    spec.task_feature_dim,
                ),
                "next_task_features": (
                    spec.count,
                    spec.max_tasks,
                    spec.task_feature_dim,
                ),
                "task_masks": (spec.count, spec.max_tasks),
                "next_task_masks": (spec.count, spec.max_tasks),
                "categorical_masks": (spec.count, spec.categorical_dim),
                "next_categorical_masks": (
                    spec.count,
                    spec.categorical_dim,
                ),
                "proposed_actions": (spec.count, spec.action_dim),
                "executed_actions": (spec.count, spec.action_dim),
            }
            for field, shape in expected.items():
                value = getattr(transition, field)[name]
                if np.asarray(value).shape != shape:
                    raise ValueError(f"{field}[{name!r}] must have shape {shape}")
            if name not in transition.rewards:
                raise ValueError(f"missing reward for role {name!r}")

    def _collate(self, transitions: Sequence[HybridTransition]) -> dict:
        batch: dict = {
            "state": self._tensor(np.stack([item.state for item in transitions])),
            "next_state": self._tensor(
                np.stack([item.next_state for item in transitions])
            ),
            "done": self._tensor(
                np.asarray([item.done for item in transitions], dtype=np.float32)
            ),
        }
        mapping_fields = (
            "observations",
            "next_observations",
            "task_features",
            "next_task_features",
            "proposed_actions",
            "executed_actions",
        )
        mask_fields = (
            "task_masks",
            "next_task_masks",
            "categorical_masks",
            "next_categorical_masks",
        )
        for field in mapping_fields:
            batch[field] = {
                spec.name: self._tensor(
                    np.stack([getattr(item, field)[spec.name] for item in transitions])
                )
                for spec in self.role_specs
            }
        for field in mask_fields:
            batch[field] = {
                spec.name: self._bool_tensor(
                    np.stack([getattr(item, field)[spec.name] for item in transitions])
                )
                for spec in self.role_specs
            }
        batch["rewards"] = {
            spec.name: self._tensor(
                np.asarray(
                    [item.rewards[spec.name] for item in transitions],
                    dtype=np.float32,
                )
            )
            for spec in self.role_specs
        }
        batch["executed_joint"] = self._flatten_joint(
            batch["executed_actions"]
        )
        return batch

    def _flatten_joint(
        self, actions: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(
            [actions[spec.name].flatten(start_dim=1) for spec in self.role_specs],
            dim=-1,
        )

    def _tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _bool_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.bool, device=self.device)

    @staticmethod
    def _gradient_norm(module: nn.Module) -> float:
        total = torch.zeros((), device=next(module.parameters()).device)
        for parameter in module.parameters():
            if parameter.grad is not None:
                total = total + parameter.grad.detach().pow(2).sum()
        return float(total.sqrt().cpu())

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)
