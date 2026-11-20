"""On-policy hybrid MAPPO baseline for the queue-aware VEC environment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from .networks import MaskedTaskSetEncoder, RoleSpec


@dataclass(frozen=True)
class HybridMAPPOConfig:
    state_dim: int
    role_specs: tuple[RoleSpec, ...]
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    actor_lr: float = 3.0e-4
    critic_lr: float = 1.0e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    repair_coefficient: float = 0.05
    ppo_epochs: int = 4
    mini_batches: int = 4
    max_gradient_norm: float = 10.0
    target_kl: float = 0.03
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or not self.role_specs:
            raise ValueError("MAPPO requires state and role specifications")
        names = [spec.name for spec in self.role_specs]
        if len(names) != len(set(names)):
            raise ValueError("MAPPO role names must be unique")
        if self.actor_hidden_dim <= 0 or self.critic_hidden_dim <= 0:
            raise ValueError("MAPPO hidden dimensions must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("invalid discount or GAE parameter")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must lie in (0, 1)")
        if self.ppo_epochs <= 0 or self.mini_batches <= 0:
            raise ValueError("PPO update counts must be positive")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        bounded_nonnegative = (
            self.entropy_coefficient,
            self.value_coefficient,
            self.repair_coefficient,
            self.max_gradient_norm,
            self.target_kl,
        )
        if any(not isfinite(value) or value < 0.0 for value in bounded_nonnegative):
            raise ValueError("MAPPO coefficients must be finite and non-negative")
        if self.max_gradient_norm == 0.0 or self.target_kl == 0.0:
            raise ValueError("gradient norm and target KL must be positive")


@dataclass(frozen=True)
class MAPPORolloutStep:
    state: np.ndarray
    observations: Mapping[str, np.ndarray]
    task_features: Mapping[str, np.ndarray]
    task_masks: Mapping[str, np.ndarray]
    categorical_masks: Mapping[str, np.ndarray]
    latent_actions: Mapping[str, np.ndarray]
    proposed_actions: Mapping[str, np.ndarray]
    executed_actions: Mapping[str, np.ndarray]
    log_probabilities: Mapping[str, np.ndarray]
    value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class MAPPOUpdateDiagnostics:
    actor_loss: Mapping[str, float]
    repair_loss: Mapping[str, float]
    entropy: Mapping[str, float]
    critic_loss: float
    approximate_kl: Mapping[str, float]
    epochs_completed: int


class HybridMAPPOActor(nn.Module):
    """Role-shared distribution over categorical and transformed latent actions."""

    def __init__(self, spec: RoleSpec, hidden_dim: int = 128) -> None:
        super().__init__()
        self.spec = spec
        self.task_encoder = MaskedTaskSetEncoder(spec.task_feature_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(spec.observation_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.categorical = (
            nn.Linear(hidden_dim, spec.categorical_dim)
            if spec.categorical_dim
            else None
        )
        self.continuous_dim = (
            int(spec.has_split)
            + int(spec.has_cpu)
            + (3 if spec.has_motion else 0)
            + (spec.max_tasks if spec.has_cpu else 0)
        )
        self.continuous_mean = (
            nn.Linear(hidden_dim, self.continuous_dim)
            if self.continuous_dim
            else None
        )
        self.continuous_log_std = (
            nn.Parameter(torch.full((self.continuous_dim,), -0.5))
            if self.continuous_dim
            else None
        )

    @property
    def latent_dim(self) -> int:
        return int(bool(self.spec.categorical_dim)) + self.continuous_dim

    def sample(
        self,
        observations: torch.Tensor,
        task_features: torch.Tensor,
        task_masks: torch.Tensor,
        categorical_masks: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        categorical_distribution, normal = self._distributions(
            observations, task_features, task_masks, categorical_masks
        )
        latent_parts: list[torch.Tensor] = []
        log_probability = torch.zeros(
            observations.shape[:-1], device=observations.device
        )
        if categorical_distribution is not None:
            category = (
                categorical_distribution.logits.argmax(dim=-1)
                if deterministic
                else categorical_distribution.sample()
            )
            latent_parts.append(category.to(observations.dtype).unsqueeze(-1))
            log_probability = log_probability + categorical_distribution.log_prob(
                category
            )
        if normal is not None:
            raw = normal.mean if deterministic else normal.sample()
            latent_parts.append(raw)
            log_probability = log_probability + self._continuous_log_probability(
                normal, raw, task_masks
            )
        latent = torch.cat(latent_parts, dim=-1)
        action = self.action_from_latent(
            latent, task_masks, categorical_masks, relaxed_categorical=False
        )
        return action, latent, log_probability

    def evaluate_latent(
        self,
        observations: torch.Tensor,
        task_features: torch.Tensor,
        task_masks: torch.Tensor,
        categorical_masks: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        categorical_distribution, normal = self._distributions(
            observations, task_features, task_masks, categorical_masks
        )
        cursor = 0
        log_probability = torch.zeros(
            observations.shape[:-1], device=observations.device
        )
        entropy = torch.zeros_like(log_probability)
        if categorical_distribution is not None:
            category = latent[..., 0].long()
            cursor = 1
            log_probability = log_probability + categorical_distribution.log_prob(
                category
            )
            entropy = entropy + categorical_distribution.entropy()
        if normal is not None:
            raw = latent[..., cursor:]
            log_probability = log_probability + self._continuous_log_probability(
                normal, raw, task_masks
            )
            entropy = entropy + self._continuous_entropy(normal, task_masks)
        mean_latent: list[torch.Tensor] = []
        if categorical_distribution is not None:
            mean_latent.append(
                categorical_distribution.logits.argmax(dim=-1).to(observations.dtype).unsqueeze(-1)
            )
        if normal is not None:
            mean_latent.append(normal.mean)
        mean_action = self.action_from_latent(
            torch.cat(mean_latent, dim=-1),
            task_masks,
            categorical_masks,
            relaxed_categorical=True,
            categorical_logits=(
                categorical_distribution.logits
                if categorical_distribution is not None
                else None
            ),
        )
        return log_probability, entropy, mean_action

    def action_from_latent(
        self,
        latent: torch.Tensor,
        task_masks: torch.Tensor,
        categorical_masks: torch.Tensor,
        *,
        relaxed_categorical: bool,
        categorical_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spec = self.spec
        latent_cursor = 0
        continuous_cursor = int(bool(spec.categorical_dim))
        fields: list[torch.Tensor] = []
        if spec.categorical_dim:
            if relaxed_categorical:
                if categorical_logits is None:
                    raise ValueError("relaxed categorical action requires logits")
                fields.append(torch.softmax(categorical_logits, dim=-1))
            else:
                category = latent[..., latent_cursor].long()
                fields.append(F.one_hot(category, spec.categorical_dim).to(latent.dtype))
            latent_cursor += 1
        cursor = continuous_cursor
        if spec.has_split:
            fields.append(torch.sigmoid(latent[..., cursor : cursor + 1]))
            cursor += 1
        if spec.has_cpu:
            fields.append(torch.sigmoid(latent[..., cursor : cursor + 1]))
            cursor += 1
        if spec.has_motion:
            fields.append(torch.tanh(latent[..., cursor : cursor + 3]))
            cursor += 3
        if spec.has_cpu:
            logits = latent[..., cursor : cursor + spec.max_tasks].masked_fill(
                ~task_masks.bool(), -1.0e9
            )
            shares = torch.softmax(logits, dim=-1)
            any_task = task_masks.any(dim=-1, keepdim=True)
            fields.append(torch.where(any_task, shares, torch.zeros_like(shares)))
        return torch.cat(fields, dim=-1)

    def _distributions(
        self,
        observations: torch.Tensor,
        task_features: torch.Tensor,
        task_masks: torch.Tensor,
        categorical_masks: torch.Tensor,
    ) -> tuple[Categorical | None, Normal | None]:
        pooled = self.task_encoder(task_features, task_masks)
        hidden = self.trunk(torch.cat((observations, pooled), dim=-1))
        categorical_distribution = None
        if self.categorical is not None:
            logits = self.categorical(hidden).masked_fill(
                ~categorical_masks.bool(), -1.0e9
            )
            categorical_distribution = Categorical(logits=logits)
        normal = None
        if self.continuous_mean is not None and self.continuous_log_std is not None:
            mean = self.continuous_mean(hidden)
            standard_deviation = self.continuous_log_std.clamp(-5.0, 2.0).exp()
            normal = Normal(mean, standard_deviation)
        return categorical_distribution, normal

    def _continuous_mask(self, task_masks: torch.Tensor) -> torch.Tensor:
        spec = self.spec
        fixed = (
            int(spec.has_split) + int(spec.has_cpu) + (3 if spec.has_motion else 0)
        )
        pieces = [
            torch.ones(
                (*task_masks.shape[:-1], fixed),
                dtype=torch.bool,
                device=task_masks.device,
            )
        ]
        if spec.has_cpu:
            pieces.append(task_masks.bool())
        return torch.cat(pieces, dim=-1)

    def _continuous_log_probability(
        self, normal: Normal, raw: torch.Tensor, task_masks: torch.Tensor
    ) -> torch.Tensor:
        mask = self._continuous_mask(task_masks).to(raw.dtype)
        return (normal.log_prob(raw) * mask).sum(dim=-1)

    def _continuous_entropy(
        self, normal: Normal, task_masks: torch.Tensor
    ) -> torch.Tensor:
        mask = self._continuous_mask(task_masks).to(normal.mean.dtype)
        return (normal.entropy() * mask).sum(dim=-1)


class CentralizedValue(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class HybridMAPPO:
    """Role-shared actors with a centralized state-value critic and GAE."""

    algorithm = "hybrid_mappo"

    def __init__(self, config: HybridMAPPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self._np_random = np.random.default_rng(config.seed)
        self.actors = nn.ModuleDict(
            {
                spec.name: HybridMAPPOActor(spec, config.actor_hidden_dim)
                for spec in config.role_specs
            }
        ).to(self.device)
        self.critic = CentralizedValue(
            config.state_dim, config.critic_hidden_dim
        ).to(self.device)
        self.actor_optimizers = {
            name: torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
            for name, actor in self.actors.items()
        }
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        self.update_count = 0

    @property
    def role_specs(self) -> tuple[RoleSpec, ...]:
        return self.config.role_specs

    def value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return float(self.critic(tensor.unsqueeze(0)).item())

    def act(
        self,
        observations: Mapping[str, np.ndarray],
        task_features: Mapping[str, np.ndarray],
        task_masks: Mapping[str, np.ndarray],
        categorical_masks: Mapping[str, np.ndarray],
        *,
        deterministic: bool = False,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        actions: dict[str, np.ndarray] = {}
        latents: dict[str, np.ndarray] = {}
        log_probabilities: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for spec in self.role_specs:
                name = spec.name
                action, latent, log_probability = self.actors[name].sample(
                    self._tensor(observations[name]),
                    self._tensor(task_features[name]),
                    self._bool_tensor(task_masks[name]),
                    self._bool_tensor(categorical_masks[name]),
                    deterministic=deterministic,
                )
                actions[name] = action.cpu().numpy()
                latents[name] = latent.cpu().numpy()
                log_probabilities[name] = log_probability.cpu().numpy()
        return actions, latents, log_probabilities

    def policy_actions(
        self,
        observations: Mapping[str, np.ndarray],
        task_features: Mapping[str, np.ndarray],
        task_masks: Mapping[str, np.ndarray],
        categorical_masks: Mapping[str, np.ndarray],
        *,
        deterministic: bool = True,
    ) -> dict[str, np.ndarray]:
        return self.act(
            observations,
            task_features,
            task_masks,
            categorical_masks,
            deterministic=deterministic,
        )[0]

    def update(
        self, rollout: Sequence[MAPPORolloutStep]
    ) -> MAPPOUpdateDiagnostics:
        if len(rollout) < 2:
            raise ValueError("MAPPO update requires at least two rollout steps")
        advantages, returns = self._gae(rollout)
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1.0e-8
        )
        states = self._tensor(np.stack([step.state for step in rollout]))
        old_values = self._tensor(np.asarray([step.value for step in rollout]))
        count = len(rollout)
        mini_batch_size = max(1, count // self.config.mini_batches)
        actor_totals = {spec.name: 0.0 for spec in self.role_specs}
        repair_totals = {spec.name: 0.0 for spec in self.role_specs}
        entropy_totals = {spec.name: 0.0 for spec in self.role_specs}
        kl_totals = {spec.name: 0.0 for spec in self.role_specs}
        critic_total = 0.0
        updates = 0
        epochs_completed = 0
        for _ in range(self.config.ppo_epochs):
            permutation = self._np_random.permutation(count)
            epoch_kl = []
            for start in range(0, count, mini_batch_size):
                indices_np = permutation[start : start + mini_batch_size]
                indices = torch.as_tensor(
                    indices_np, dtype=torch.long, device=self.device
                )
                selected_advantages = advantages[indices]
                for spec in self.role_specs:
                    name = spec.name
                    observations = self._tensor(
                        np.stack([rollout[index].observations[name] for index in indices_np])
                    )
                    features = self._tensor(
                        np.stack([rollout[index].task_features[name] for index in indices_np])
                    )
                    task_masks = self._bool_tensor(
                        np.stack([rollout[index].task_masks[name] for index in indices_np])
                    )
                    categorical_masks = self._bool_tensor(
                        np.stack(
                            [rollout[index].categorical_masks[name] for index in indices_np]
                        )
                    )
                    latent = self._tensor(
                        np.stack([rollout[index].latent_actions[name] for index in indices_np])
                    )
                    old_log_probability = self._tensor(
                        np.stack(
                            [rollout[index].log_probabilities[name] for index in indices_np]
                        )
                    )
                    executed = self._tensor(
                        np.stack([rollout[index].executed_actions[name] for index in indices_np])
                    )
                    new_log_probability, entropy, mean_action = self.actors[
                        name
                    ].evaluate_latent(
                        observations,
                        features,
                        task_masks,
                        categorical_masks,
                        latent,
                    )
                    role_advantages = selected_advantages[:, None].expand(
                        -1, spec.count
                    )
                    ratio = torch.exp(
                        (new_log_probability - old_log_probability).clamp(-10.0, 10.0)
                    )
                    unclipped = ratio * role_advantages
                    clipped = ratio.clamp(
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * role_advantages
                    actor_loss = -torch.minimum(unclipped, clipped).mean()
                    repair_loss = F.mse_loss(mean_action, executed)
                    total_actor = (
                        actor_loss
                        - self.config.entropy_coefficient * entropy.mean()
                        + self.config.repair_coefficient * repair_loss
                    )
                    optimizer = self.actor_optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    total_actor.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[name].parameters(),
                        self.config.max_gradient_norm,
                    )
                    optimizer.step()
                    with torch.no_grad():
                        approximate_kl = (
                            (ratio - 1.0) - torch.log(ratio.clamp_min(1.0e-12))
                        ).mean()
                    actor_totals[name] += float(actor_loss.detach().cpu())
                    repair_totals[name] += float(repair_loss.detach().cpu())
                    entropy_totals[name] += float(entropy.mean().detach().cpu())
                    kl_value = float(approximate_kl.cpu())
                    kl_totals[name] += kl_value
                    epoch_kl.append(kl_value)

                predicted = self.critic(states[indices])
                value_delta = predicted - old_values[indices]
                clipped_values = old_values[indices] + value_delta.clamp(
                    -self.config.clip_epsilon, self.config.clip_epsilon
                )
                value_loss = torch.maximum(
                    (predicted - returns[indices]).pow(2),
                    (clipped_values - returns[indices]).pow(2),
                ).mean()
                critic_loss = self.config.value_coefficient * value_loss
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_gradient_norm
                )
                self.critic_optimizer.step()
                critic_total += float(critic_loss.detach().cpu())
                updates += 1
            epochs_completed += 1
            if epoch_kl and max(epoch_kl) > self.config.target_kl:
                break
        self.update_count += 1
        denominator = max(1, updates)
        return MAPPOUpdateDiagnostics(
            actor_loss={name: value / denominator for name, value in actor_totals.items()},
            repair_loss={name: value / denominator for name, value in repair_totals.items()},
            entropy={name: value / denominator for name, value in entropy_totals.items()},
            critic_loss=critic_total / denominator,
            approximate_kl={name: value / denominator for name, value in kl_totals.items()},
            epochs_completed=epochs_completed,
        )

    def fork(self) -> "HybridMAPPO":
        torch_rng_state = torch.get_rng_state()
        clone = HybridMAPPO(self.config)
        torch.set_rng_state(torch_rng_state)
        clone.actors.load_state_dict(deepcopy(self.actors.state_dict()))
        clone.critic.load_state_dict(deepcopy(self.critic.state_dict()))
        clone.update_count = self.update_count
        return clone

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "algorithm": self.algorithm,
                "config": self.config,
                "actors": self.actors.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizers": {
                    name: optimizer.state_dict()
                    for name, optimizer in self.actor_optimizers.items()
                },
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self.update_count,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": deepcopy(self._np_random.bit_generator.state),
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != self.algorithm:
            raise ValueError("checkpoint is not a hybrid MAPPO checkpoint")
        if checkpoint["config"] != self.config:
            raise ValueError("checkpoint configuration does not match learner")
        self.actors.load_state_dict(checkpoint["actors"])
        self.critic.load_state_dict(checkpoint["critic"])
        for name, state in checkpoint["actor_optimizers"].items():
            self.actor_optimizers[name].load_state_dict(state)
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint["update_count"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        self._np_random.bit_generator.state = checkpoint["numpy_rng_state"]

    def _gae(
        self, rollout: Sequence[MAPPORolloutStep]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = np.zeros(len(rollout), dtype=np.float32)
        accumulator = 0.0
        next_value = 0.0
        for index in range(len(rollout) - 1, -1, -1):
            step = rollout[index]
            continuation = 0.0 if step.done else 1.0
            delta = (
                step.reward
                + self.config.gamma * next_value * continuation
                - step.value
            )
            accumulator = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * continuation
                * accumulator
            )
            advantages[index] = accumulator
            next_value = step.value
        advantage_tensor = self._tensor(advantages)
        values = self._tensor(np.asarray([step.value for step in rollout]))
        return advantage_tensor, advantage_tensor + values

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _bool_tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.bool, device=self.device)
