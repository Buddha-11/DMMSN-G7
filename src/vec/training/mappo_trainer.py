"""Reproducible on-policy training orchestration for hybrid VEC MAPPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import platform
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping

import numpy as np
import torch

from ..evaluation.runner import code_identity
from ..evaluation.scenarios import (
    ScenarioInstance,
    ScenarioManifest,
    ScenarioSpec,
    instantiate_scenario,
)
from ..learning import (
    HybridMAPPO,
    HybridMAPPOConfig,
    MAPPORolloutStep,
    VecPolicyAdapter,
)
from .trainer import TrainingResult


MAPPO_TRAINING_SCHEMA_VERSION = "vec-mappo-training/v1"


@dataclass(frozen=True)
class MAPPOTrainingConfig:
    seed: int = 0
    epochs: int = 50
    scenarios_per_epoch: int = 7
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
    validation_every: int = 1
    validation_scenarios: int = 35
    early_stopping_patience: int = 10
    device: str = "cpu"
    schema_version: str = MAPPO_TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MAPPO_TRAINING_SCHEMA_VERSION:
            raise ValueError("unsupported MAPPO training schema")
        positive = (
            self.epochs,
            self.scenarios_per_epoch,
            self.max_tasks,
            self.actor_hidden_dim,
            self.critic_hidden_dim,
            self.ppo_epochs,
            self.mini_batches,
            self.validation_every,
            self.validation_scenarios,
            self.early_stopping_patience,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("MAPPO training counts and dimensions must be positive")
        if not self.train_stresses:
            raise ValueError("at least one training stress is required")
        object.__setattr__(self, "train_stresses", tuple(self.train_stresses))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MAPPOTrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["train_stresses"] = tuple(payload["train_stresses"])
        return cls(**payload)


class VecMAPPOTrainer:
    """Train hybrid MAPPO on train scenarios and select only on validation."""

    def __init__(
        self,
        config: MAPPOTrainingConfig,
        manifest: ScenarioManifest,
        output_directory: str | Path,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.output = Path(output_directory)
        self.random = np.random.default_rng(config.seed)
        allowed = set(config.train_stresses)
        self.training_specs = tuple(
            spec
            for spec in manifest.select(split="train")
            if spec.stress in allowed
        )
        self.validation_specs = manifest.select(split="validation")
        if not self.training_specs or not self.validation_specs:
            raise ValueError("MAPPO training requires train and validation scenarios")
        prototype = instantiate_scenario(self.training_specs[0])
        environment = prototype.environment()
        adapter = VecPolicyAdapter(environment, max_tasks=config.max_tasks)
        self.learner = HybridMAPPO(
            HybridMAPPOConfig(
                state_dim=len(
                    environment.control_observation().global_critic_state
                ),
                role_specs=adapter.role_specs(),
                actor_hidden_dim=config.actor_hidden_dim,
                critic_hidden_dim=config.critic_hidden_dim,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_epsilon=config.clip_epsilon,
                entropy_coefficient=config.entropy_coefficient,
                value_coefficient=config.value_coefficient,
                repair_coefficient=config.repair_coefficient,
                ppo_epochs=config.ppo_epochs,
                mini_batches=config.mini_batches,
                max_gradient_norm=config.max_gradient_norm,
                target_kl=config.target_kl,
                seed=config.seed,
                device=config.device,
            )
        )
        self.completed_epochs = 0
        self.best_epoch = 0
        self.best_score = -float("inf")
        self.stale_validations = 0
        self.elapsed_before_resume = 0.0

    def train(self, *, resume: bool = False) -> TrainingResult:
        self.output.mkdir(parents=True, exist_ok=True)
        self._prepare_or_resume(resume)
        started = perf_counter()
        stopped_early = False
        for epoch in range(self.completed_epochs + 1, self.config.epochs + 1):
            record = self._train_epoch(epoch)
            if epoch % self.config.validation_every == 0:
                validation = self._validate()
                record.update(validation)
                score = validation["validation_score"]
                if score is not None and float(score) > self.best_score:
                    self.best_score = float(score)
                    self.best_epoch = epoch
                    self.stale_validations = 0
                    self.learner.save(self.output / "best.pt")
                else:
                    self.stale_validations += 1
            self.completed_epochs = epoch
            elapsed = self.elapsed_before_resume + perf_counter() - started
            record["offline_training_seconds"] = elapsed
            self._append_history(record)
            self.learner.save(self.output / "last.pt")
            self._save_state(elapsed)
            if self.stale_validations >= self.config.early_stopping_patience:
                stopped_early = True
                break
        elapsed = self.elapsed_before_resume + perf_counter() - started
        self._save_state(elapsed)
        result = TrainingResult(
            output_directory=str(self.output.resolve()),
            completed_epochs=self.completed_epochs,
            best_epoch=self.best_epoch or None,
            best_validation_score=(
                self.best_score if isfinite(self.best_score) else None
            ),
            stopped_early=stopped_early,
            offline_training_seconds=elapsed,
            best_checkpoint=(
                str((self.output / "best.pt").resolve())
                if (self.output / "best.pt").exists()
                else None
            ),
            last_checkpoint=str((self.output / "last.pt").resolve()),
        )
        (self.output / "training_summary.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_artifact_manifest()
        return result

    def _train_epoch(self, epoch: int) -> dict[str, object]:
        indices = self.random.permutation(len(self.training_specs))
        chosen = [
            self.training_specs[int(index)]
            for index in indices[: self.config.scenarios_per_epoch]
        ]
        if len(chosen) < self.config.scenarios_per_epoch:
            chosen.extend(
                self.training_specs[int(index)]
                for index in self.random.integers(
                    0,
                    len(self.training_specs),
                    self.config.scenarios_per_epoch - len(chosen),
                )
            )
        rewards: list[float] = []
        critic_losses: list[float] = []
        completed_ppo_epochs: list[int] = []
        actor_losses: dict[str, list[float]] = {
            spec.name: [] for spec in self.learner.role_specs
        }
        repair_losses: dict[str, list[float]] = {
            spec.name: [] for spec in self.learner.role_specs
        }
        entropies: dict[str, list[float]] = {
            spec.name: [] for spec in self.learner.role_specs
        }
        approximate_kls: dict[str, list[float]] = {
            spec.name: [] for spec in self.learner.role_specs
        }
        for spec in chosen:
            rollout, reward = self._collect(instantiate_scenario(spec))
            diagnostics = self.learner.update(rollout)
            rewards.append(reward)
            critic_losses.append(diagnostics.critic_loss)
            completed_ppo_epochs.append(diagnostics.epochs_completed)
            for role in actor_losses:
                actor_losses[role].append(diagnostics.actor_loss[role])
                repair_losses[role].append(diagnostics.repair_loss[role])
                entropies[role].append(diagnostics.entropy[role])
                approximate_kls[role].append(diagnostics.approximate_kl[role])
        return {
            "epoch": epoch,
            "mode": "hybrid_mappo",
            "scenario_ids": [spec.scenario_id for spec in chosen],
            "mean_training_reward_per_slot": float(np.mean(rewards)),
            "mean_critic_loss": float(np.mean(critic_losses)),
            "mean_ppo_epochs_completed": float(np.mean(completed_ppo_epochs)),
            "mean_actor_loss": {
                role: float(np.mean(values))
                for role, values in actor_losses.items()
            },
            "mean_repair_loss": {
                role: float(np.mean(values))
                for role, values in repair_losses.items()
            },
            "mean_entropy": {
                role: float(np.mean(values))
                for role, values in entropies.items()
            },
            "mean_approximate_kl": {
                role: float(np.mean(values))
                for role, values in approximate_kls.items()
            },
            "policy_updates": len(chosen),
            "learner_update_count": self.learner.update_count,
        }

    def _collect(
        self, scenario: ScenarioInstance
    ) -> tuple[list[MAPPORolloutStep], float]:
        environment = scenario.environment()
        adapter = VecPolicyAdapter(environment, max_tasks=self.config.max_tasks)
        rollout: list[MAPPORolloutStep] = []
        cumulative_reward = 0.0
        total_slots = scenario.spec.horizon_slots + scenario.spec.drain_slots
        for _ in range(total_slots):
            encoded = adapter.encode()
            actions, latent, log_probabilities = self.learner.act(
                encoded.observations,
                encoded.task_features,
                encoded.task_masks,
                encoded.categorical_masks,
                deterministic=False,
            )
            value = self.learner.value(encoded.state)
            result = environment.step(adapter.decode(encoded, actions))
            executed = adapter.encode_executed_actions(encoded, actions, result)
            done = result.terminated or result.truncated
            rollout.append(
                MAPPORolloutStep(
                    state=encoded.state,
                    observations=encoded.observations,
                    task_features=encoded.task_features,
                    task_masks=encoded.task_masks,
                    categorical_masks=encoded.categorical_masks,
                    latent_actions=latent,
                    proposed_actions=actions,
                    executed_actions=executed,
                    log_probabilities=log_probabilities,
                    value=value,
                    reward=result.lyapunov.reward,
                    done=done,
                )
            )
            cumulative_reward += result.lyapunov.reward
            if done:
                break
        return rollout, cumulative_reward / max(1, len(rollout))

    def _validate(self) -> dict[str, object]:
        selected = self.validation_specs[: self.config.validation_scenarios]
        rewards: list[float] = []
        completion_by_scenario: dict[str, float] = {}
        reward_by_scenario: dict[str, float] = {}
        violations_by_scenario: dict[str, int] = {}
        completed = 0
        admitted = 0
        violations = 0
        for spec in selected:
            learner = self.learner.fork()
            scenario = instantiate_scenario(spec)
            environment = scenario.environment()
            adapter = VecPolicyAdapter(environment, self.config.max_tasks)
            cumulative = 0.0
            slots = 0
            scenario_violations = 0
            for _ in range(spec.horizon_slots + spec.drain_slots):
                encoded = adapter.encode()
                actions = learner.policy_actions(
                    encoded.observations,
                    encoded.task_features,
                    encoded.task_masks,
                    encoded.categorical_masks,
                    deterministic=True,
                )
                result = environment.step(adapter.decode(encoded, actions))
                cumulative += result.lyapunov.reward
                slots += 1
                scenario_violations += int(result.physical_feasibility_failure)
                if result.terminated or result.truncated:
                    break
            scenario_reward = cumulative / max(1, slots)
            scenario_admitted = len(environment.tasks)
            scenario_completed = sum(
                record.is_complete for record in environment.tasks.values()
            )
            rewards.append(scenario_reward)
            completed += scenario_completed
            admitted += scenario_admitted
            violations += scenario_violations
            reward_by_scenario[spec.scenario_id] = scenario_reward
            completion_by_scenario[spec.scenario_id] = (
                scenario_completed / scenario_admitted
                if scenario_admitted
                else 1.0
            )
            violations_by_scenario[spec.scenario_id] = scenario_violations
        return {
            "validation_score": (
                float(np.mean(rewards)) if violations == 0 else None
            ),
            "validation_checkpoint_eligible": violations == 0,
            "validation_mean_reward_per_slot": float(np.mean(rewards)),
            "validation_completion_rate": completed / admitted if admitted else 1.0,
            "validation_physical_violations": violations,
            "validation_scenario_ids": [spec.scenario_id for spec in selected],
            "validation_reward_by_scenario": reward_by_scenario,
            "validation_completion_by_scenario": completion_by_scenario,
            "validation_violations_by_scenario": violations_by_scenario,
        }

    def _prepare_or_resume(self, resume: bool) -> None:
        config_path = self.output / "training_config.json"
        state_path = self.output / "training_state.pt"
        if resume:
            if not config_path.exists() or not state_path.exists():
                raise FileNotFoundError("resume requested but MAPPO state is missing")
            if MAPPOTrainingConfig.load(config_path) != self.config:
                raise ValueError("resume configuration differs from saved experiment")
            self.learner.load(self.output / "last.pt")
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.completed_epochs = int(state["completed_epochs"])
            self.best_epoch = int(state["best_epoch"])
            self.best_score = float(state["best_score"])
            self.stale_validations = int(state["stale_validations"])
            self.elapsed_before_resume = float(state["elapsed_seconds"])
            self.random.bit_generator.state = state["trainer_rng_state"]
            return
        if state_path.exists():
            raise FileExistsError("output already has MAPPO state; use resume=True")
        self.config.save(config_path)
        self.manifest.save(self.output / "scenario_manifest.json")
        revision, dirty = code_identity()
        (self.output / "provenance.json").write_text(
            json.dumps(
                {
                    "algorithm": "hybrid_mappo",
                    "code_revision": revision,
                    "code_dirty": dirty,
                    "test_split_used_for_training": False,
                    "checkpoint_selection_metric": "validation mean reward per slot; safety-feasible only",
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy_version": np.__version__,
                    "torch_version": torch.__version__,
                    "python_executable": sys.executable,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _save_state(self, elapsed_seconds: float) -> None:
        torch.save(
            {
                "completed_epochs": self.completed_epochs,
                "best_epoch": self.best_epoch,
                "best_score": self.best_score,
                "stale_validations": self.stale_validations,
                "elapsed_seconds": elapsed_seconds,
                "trainer_rng_state": self.random.bit_generator.state,
            },
            self.output / "training_state.pt",
        )

    def _append_history(self, record: Mapping[str, object]) -> None:
        with (self.output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _write_artifact_manifest(self) -> None:
        files = sorted(
            path
            for path in self.output.iterdir()
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        payload = {
            "schema_version": "vec-training-artifacts/v1",
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": self._file_hash(path),
                }
                for path in files
            },
        }
        (self.output / "artifact_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
