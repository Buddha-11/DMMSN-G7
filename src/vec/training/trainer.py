"""Training, validation, checkpointing, and provenance for learned policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    HybridMADDPG,
    HybridMADDPGConfig,
    ReptileMetaLearner,
    RoleSpec,
    VecPolicyAdapter,
)
from ..learning.replay import HybridTransition
from .config import TrainingConfig


@dataclass(frozen=True)
class TrainingResult:
    output_directory: str
    completed_epochs: int
    best_epoch: int | None
    best_validation_score: float | None
    stopped_early: bool
    offline_training_seconds: float
    best_checkpoint: str | None
    last_checkpoint: str


def explore_hybrid_actions(
    role_specs: tuple[RoleSpec, ...],
    actions: Mapping[str, np.ndarray],
    task_masks: Mapping[str, np.ndarray],
    categorical_masks: Mapping[str, np.ndarray],
    *,
    standard_deviation: float,
    random: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Add bounded noise while retaining every hybrid action constraint."""

    explored: dict[str, np.ndarray] = {}
    for spec in role_specs:
        values = np.asarray(actions[spec.name], dtype=np.float32).copy()
        cursor = 0
        if spec.categorical_dim:
            feasible = np.asarray(categorical_masks[spec.name], dtype=np.bool_)
            for agent in range(spec.count):
                choices = np.flatnonzero(feasible[agent])
                if len(choices) and random.random() < standard_deviation:
                    category = int(random.choice(choices))
                    values[agent, : spec.categorical_dim] = 0.0
                    values[agent, category] = 1.0
            cursor += spec.categorical_dim
        if spec.has_split:
            values[:, cursor] = np.clip(
                values[:, cursor] + random.normal(0.0, standard_deviation, spec.count),
                0.0,
                1.0,
            )
            cursor += 1
        if spec.has_cpu:
            values[:, cursor] = np.clip(
                values[:, cursor] + random.normal(0.0, standard_deviation, spec.count),
                0.0,
                1.0,
            )
            cursor += 1
        if spec.has_motion:
            values[:, cursor : cursor + 3] = np.clip(
                values[:, cursor : cursor + 3]
                + random.normal(0.0, standard_deviation, (spec.count, 3)),
                -1.0,
                1.0,
            )
            cursor += 3
        if spec.has_cpu:
            mask = np.asarray(task_masks[spec.name], dtype=np.bool_)
            shares = np.maximum(
                0.0,
                values[:, cursor : cursor + spec.max_tasks]
                + random.normal(
                    0.0, standard_deviation, (spec.count, spec.max_tasks)
                ),
            )
            shares *= mask
            totals = shares.sum(axis=1, keepdims=True)
            shares = np.divide(
                shares,
                totals,
                out=np.zeros_like(shares),
                where=totals > 0.0,
            )
            values[:, cursor : cursor + spec.max_tasks] = shares
        explored[spec.name] = values.astype(np.float32, copy=False)
    return explored


class VecTrainer:
    """Train one reproducible policy without ever looking at the test split."""

    def __init__(
        self,
        config: TrainingConfig,
        manifest: ScenarioManifest,
        output_directory: str | Path,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.output = Path(output_directory)
        self.random = np.random.default_rng(config.seed)
        self.training_specs = self._training_specs()
        self.validation_specs = manifest.select(split="validation")
        if not self.training_specs:
            raise ValueError("manifest has no compatible training scenarios")
        if not self.validation_specs:
            raise ValueError("manifest has no validation scenarios")
        prototype = self._scenario(self.training_specs[0])
        environment = prototype.environment()
        adapter = VecPolicyAdapter(environment, max_tasks=config.max_tasks)
        state_dim = len(environment.control_observation().global_critic_state)
        self.learner = HybridMADDPG(
            HybridMADDPGConfig(
                state_dim=state_dim,
                role_specs=adapter.role_specs(),
                actor_hidden_dim=config.actor_hidden_dim,
                critic_hidden_dim=config.critic_hidden_dim,
                actor_lr=config.actor_lr,
                critic_lr=config.critic_lr,
                gamma=config.gamma,
                tau=config.tau,
                batch_size=config.batch_size,
                replay_capacity=config.replay_capacity,
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
            noise = self.config.exploration_std * (
                self.config.exploration_decay ** (epoch - 1)
            )
            record = self._train_epoch(epoch, noise)
            if epoch % self.config.validation_every == 0:
                validation = self._validate()
                record.update(validation)
                raw_score = validation["validation_score"]
                if raw_score is not None and float(raw_score) > self.best_score:
                    score = float(raw_score)
                    self.best_score = score
                    self.best_epoch = epoch
                    self.stale_validations = 0
                    self.learner.save(self.output / "best.pt")
                else:
                    self.stale_validations += 1
            self.completed_epochs = epoch
            total_elapsed = self.elapsed_before_resume + perf_counter() - started
            record["offline_training_seconds"] = total_elapsed
            self._append_history(record)
            self.learner.save(self.output / "last.pt")
            self._save_state(total_elapsed)
            if self.stale_validations >= self.config.early_stopping_patience:
                stopped_early = True
                break
        total_elapsed = self.elapsed_before_resume + perf_counter() - started
        # Persist the same final elapsed time reported in the public summary.
        # This keeps a no-op resume monotonic instead of reverting to the
        # earlier per-epoch timestamp saved before checkpoint serialization.
        self._save_state(total_elapsed)
        result = TrainingResult(
            output_directory=str(self.output.resolve()),
            completed_epochs=self.completed_epochs,
            best_epoch=self.best_epoch or None,
            best_validation_score=(
                self.best_score if isfinite(self.best_score) else None
            ),
            stopped_early=stopped_early,
            offline_training_seconds=total_elapsed,
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

    def _training_specs(self) -> tuple[ScenarioSpec, ...]:
        train = self.manifest.select(split="train")
        if self.config.mode == "maddpg":
            return tuple(item for item in train if item.stress == self.config.base_stress)
        allowed = set(self.config.train_stresses)
        return tuple(item for item in train if item.stress in allowed)

    def _scenario(
        self, spec: ScenarioSpec, *, training_objective: bool = False
    ) -> ScenarioInstance:
        scenario = instantiate_scenario(spec)
        if not (training_objective and self.config.no_queue_weight):
            return scenario
        lyapunov = replace(scenario.config.lyapunov, physical_queue_weight=0.0)
        return replace(scenario, config=replace(scenario.config, lyapunov=lyapunov))

    def _train_epoch(self, epoch: int, noise: float) -> dict[str, object]:
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
        updates = 0
        if self.config.mode == "reptile":
            meta = ReptileMetaLearner(
                self.learner, self.config.reptile_outer_step_size
            )
            adapted = []
            for spec in chosen:
                collector = self.learner.fork()
                transitions, reward = self._collect_episode(
                    collector,
                    self._scenario(spec, training_objective=True),
                    noise,
                    update_online=False,
                )
                result = meta.adapt_task(
                    spec.scenario_id,
                    transitions,
                    self.config.reptile_inner_updates,
                )
                adapted.append(result.learner)
                rewards.append(reward)
                updates += len(result.diagnostics)
            meta.outer_update(adapted)
        else:
            if self.learner.current_task_id is None:
                self.learner.begin_meta_task(
                    f"{self.config.mode}:shared-training-domain"
                )
            for spec in chosen:
                _, reward, episode_updates = self._collect_episode(
                    self.learner,
                    self._scenario(spec, training_objective=True),
                    noise,
                    update_online=True,
                    return_updates=True,
                )
                rewards.append(reward)
                updates += episode_updates
        return {
            "epoch": epoch,
            "mode": self.config.mode,
            "scenario_ids": [item.scenario_id for item in chosen],
            "exploration_std": noise,
            "mean_training_reward_per_slot": float(np.mean(rewards)),
            "gradient_updates": updates,
            "learner_update_count": self.learner.update_count,
        }

    def _collect_episode(
        self,
        learner: HybridMADDPG,
        scenario: ScenarioInstance,
        noise: float,
        *,
        update_online: bool,
        return_updates: bool = False,
    ):
        if not update_online:
            learner.begin_meta_task(scenario.spec.scenario_id)
        environment = scenario.environment()
        adapter = VecPolicyAdapter(environment, max_tasks=self.config.max_tasks)
        transitions: list[HybridTransition] = []
        cumulative_reward = 0.0
        updates = 0
        total_slots = scenario.spec.horizon_slots + scenario.spec.drain_slots
        for _ in range(total_slots):
            before = adapter.encode()
            proposal = learner.act(
                before.observations,
                before.task_features,
                before.task_masks,
                before.categorical_masks,
                deterministic=False,
            )
            proposal = explore_hybrid_actions(
                learner.role_specs,
                proposal,
                before.task_masks,
                before.categorical_masks,
                standard_deviation=noise,
                random=self.random,
            )
            result = environment.step(adapter.decode(before, proposal))
            after = adapter.encode(result.observation_after, result.masks_after)
            transition = adapter.transition(before, after, proposal, result)
            transitions.append(transition)
            learner.remember(transition)
            cumulative_reward += result.lyapunov.reward
            replay_ready = (
                learner.current_task_id is not None
                and learner.replay.size(learner.current_task_id)
                >= max(self.config.warmup_steps, learner.config.batch_size)
            )
            if update_online and replay_ready:
                for _ in range(self.config.updates_per_step):
                    learner.update()
                    updates += 1
            if result.terminated or result.truncated:
                break
        mean_reward = cumulative_reward / max(1, len(transitions))
        if return_updates:
            return transitions, mean_reward, updates
        return transitions, mean_reward

    def _validate(self) -> dict[str, object]:
        selected = self.validation_specs[: self.config.validation_scenarios]
        rewards: list[float] = []
        violations = 0
        completed = 0
        admitted = 0
        reward_by_scenario: dict[str, float] = {}
        completion_by_scenario: dict[str, float] = {}
        violations_by_scenario: dict[str, int] = {}
        for spec in selected:
            learner = self.learner.fork()
            scenario = self._scenario(spec)
            learner.begin_meta_task(f"validation:{spec.scenario_id}")
            environment = scenario.environment()
            adapter = VecPolicyAdapter(environment, self.config.max_tasks)
            cumulative = 0.0
            slots = 0
            scenario_violations = 0
            for _ in range(spec.horizon_slots + spec.drain_slots):
                encoded = adapter.encode()
                actions = learner.act(
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
            violations += scenario_violations
            admitted += scenario_admitted
            completed += scenario_completed
            reward_by_scenario[spec.scenario_id] = scenario_reward
            completion_by_scenario[spec.scenario_id] = (
                scenario_completed / scenario_admitted
                if scenario_admitted
                else 1.0
            )
            violations_by_scenario[spec.scenario_id] = scenario_violations
        # Safety violations make a checkpoint ineligible, rather than allowing
        # a reward trade-off to hide an invalid controller.
        score = float(np.mean(rewards)) if violations == 0 else None
        return {
            "validation_score": score,
            "validation_checkpoint_eligible": violations == 0,
            "validation_mean_reward_per_slot": float(np.mean(rewards)),
            "validation_completion_rate": completed / admitted if admitted else 1.0,
            "validation_physical_violations": violations,
            "validation_scenario_ids": [item.scenario_id for item in selected],
            "validation_reward_by_scenario": reward_by_scenario,
            "validation_completion_by_scenario": completion_by_scenario,
            "validation_violations_by_scenario": violations_by_scenario,
        }

    def _prepare_or_resume(self, resume: bool) -> None:
        config_path = self.output / "training_config.json"
        manifest_path = self.output / "scenario_manifest.json"
        state_path = self.output / "training_state.pt"
        if resume:
            if not config_path.exists() or not state_path.exists():
                raise FileNotFoundError("resume requested but training state is missing")
            if TrainingConfig.load(config_path) != self.config:
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
            raise FileExistsError(
                "output directory already contains training state; use resume=True"
            )
        self.config.save(config_path)
        self.manifest.save(manifest_path)
        revision, dirty = code_identity()
        (self.output / "provenance.json").write_text(
            json.dumps(
                {
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
