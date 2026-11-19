"""Role-shared hybrid actors and centralized critics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RoleSpec:
    name: str
    count: int
    observation_dim: int
    task_feature_dim: int
    max_tasks: int
    categorical_dim: int = 0
    has_split: bool = False
    has_cpu: bool = False
    has_motion: bool = False

    @property
    def action_dim(self) -> int:
        return (
            self.categorical_dim
            + int(self.has_split)
            + int(self.has_cpu)
            + (3 if self.has_motion else 0)
            + (self.max_tasks if self.has_cpu else 0)
        )


class MaskedTaskSetEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = hidden_dim
        self.item = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, tasks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.item(tasks)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        total = (encoded * weights).sum(dim=-2)
        count = weights.sum(dim=-2).clamp_min(1.0)
        return total / count


class HybridRoleActor(nn.Module):
    """One shared actor for every agent belonging to a role."""

    def __init__(self, spec: RoleSpec, hidden_dim: int = 128) -> None:
        super().__init__()
        self.spec = spec
        self.task_encoder = MaskedTaskSetEncoder(
            spec.task_feature_dim, hidden_dim
        )
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
        self.split = nn.Linear(hidden_dim, 1) if spec.has_split else None
        self.cpu_head = nn.Linear(hidden_dim, 1) if spec.has_cpu else None
        self.motion = nn.Linear(hidden_dim, 3) if spec.has_motion else None
        self.shares = (
            nn.Linear(hidden_dim, spec.max_tasks) if spec.has_cpu else None
        )

    def forward(
        self,
        observation: torch.Tensor,
        task_features: torch.Tensor,
        task_mask: torch.Tensor,
        categorical_mask: torch.Tensor | None = None,
        *,
        temperature: float = 1.0,
        hard: bool = True,
        deterministic: bool = False,
    ) -> torch.Tensor:
        pooled = self.task_encoder(task_features, task_mask)
        hidden = self.trunk(torch.cat((observation, pooled), dim=-1))
        fields: list[torch.Tensor] = []
        if self.categorical is not None:
            logits = self.categorical(hidden)
            if categorical_mask is not None:
                logits = logits.masked_fill(~categorical_mask.bool(), -1.0e9)
            if deterministic:
                index = logits.argmax(dim=-1)
                categorical = F.one_hot(
                    index, num_classes=self.spec.categorical_dim
                ).to(logits.dtype)
            else:
                categorical = F.gumbel_softmax(
                    logits, tau=temperature, hard=hard, dim=-1
                )
            fields.append(categorical)
        if self.split is not None:
            fields.append(torch.sigmoid(self.split(hidden)))
        if self.cpu_head is not None:
            fields.append(torch.sigmoid(self.cpu_head(hidden)))
        if self.motion is not None:
            fields.append(torch.tanh(self.motion(hidden)))
        if self.shares is not None:
            share_logits = self.shares(hidden).masked_fill(
                ~task_mask.bool(), -1.0e9
            )
            shares = torch.softmax(share_logits, dim=-1)
            any_task = task_mask.any(dim=-1, keepdim=True)
            fields.append(torch.where(any_task, shares, torch.zeros_like(shares)))
        return torch.cat(fields, dim=-1)


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, joint_action), dim=-1)).squeeze(-1)


class HybridActionRepair:
    """Deterministic tensor repair used for replay and target actions."""

    def repair(
        self,
        spec: RoleSpec,
        action: torch.Tensor,
        task_mask: torch.Tensor,
        categorical_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cursor = 0
        fields: list[torch.Tensor] = []
        if spec.categorical_dim:
            values = action[..., : spec.categorical_dim]
            cursor += spec.categorical_dim
            if categorical_mask is not None:
                values = values.masked_fill(~categorical_mask.bool(), -1.0e9)
            index = values.argmax(dim=-1)
            fields.append(
                F.one_hot(index, spec.categorical_dim).to(action.dtype)
            )
        if spec.has_split:
            fields.append(action[..., cursor : cursor + 1].clamp(0.0, 1.0))
            cursor += 1
        if spec.has_cpu:
            fields.append(action[..., cursor : cursor + 1].clamp(0.0, 1.0))
            cursor += 1
        if spec.has_motion:
            fields.append(action[..., cursor : cursor + 3].clamp(-1.0, 1.0))
            cursor += 3
        if spec.has_cpu:
            shares = action[..., cursor : cursor + spec.max_tasks].clamp_min(0.0)
            shares = shares * task_mask.to(action.dtype)
            denominator = shares.sum(dim=-1, keepdim=True)
            shares = torch.where(
                denominator > 0,
                shares / denominator.clamp_min(1.0e-12),
                torch.zeros_like(shares),
            )
            fields.append(shares)
        return torch.cat(fields, dim=-1)
