"""Paired-scenario evaluation runner shared by every controller family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import subprocess
from time import perf_counter
from typing import Protocol

from ..baselines import (
    BaselineController,
    GeneticDriftController,
    LocalOnlyController,
    NearestFeasibleEdgeController,
    QueueEnergyGreedyController,
    RandomFeasibleController,
)
from ..env import VecQueueEnv
from .metrics import RunMetrics, SlotTrace, TaskOutcome, make_slot_trace, summarize_run
from .scenarios import ScenarioInstance


REQUIRED_COMPARISON_METHODS = (
    "local_only",
    "nearest_feasible_edge",
    "queue_energy_greedy",
    "genetic_drift",
    "hybrid_maddpg",
    "domain_randomized_maddpg",
    "reptile_maddpg",
    "reptile_no_queue_weight",
    "hybrid_mappo",
)


class ControllerFactory(Protocol):
    def __call__(
        self, scenario: ScenarioInstance, environment: VecQueueEnv
    ) -> BaselineController:
        ...


@dataclass(frozen=True)
class EvaluationMethod:
    name: str
    family: str
    factory: ControllerFactory
    adaptation_updates: int = 0
    offline_training_seconds: float | None = None
    checkpoint: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation method name cannot be empty")
        if self.adaptation_updates < 0:
            raise ValueError("adaptation_updates cannot be negative")
        if self.offline_training_seconds is not None and self.offline_training_seconds < 0:
            raise ValueError("offline training time cannot be negative")


@dataclass(frozen=True)
class EvaluationRun:
    schema_version: str
    scenario_id: str
    split: str
    stress: str
    seed: int
    method: str
    method_family: str
    trace_digest: str
    configuration_digest: str
    code_revision: str
    code_dirty: bool
    adaptation_updates_budget: int
    checkpoint: str | None
    metrics: RunMetrics
    traces: tuple[SlotTrace, ...]
    tasks: tuple[TaskOutcome, ...]


def baseline_methods(
    *, genetic_budget: int = 64
) -> tuple[EvaluationMethod, ...]:
    """Reference methods; learned methods are registered with checkpoints."""

    return (
        EvaluationMethod(
            "local_only",
            "non_learning",
            lambda scenario, environment: LocalOnlyController(),
        ),
        EvaluationMethod(
            "nearest_feasible_edge",
            "non_learning",
            lambda scenario, environment: NearestFeasibleEdgeController(),
        ),
        EvaluationMethod(
            "queue_energy_greedy",
            "non_learning",
            lambda scenario, environment: QueueEnergyGreedyController(),
        ),
        EvaluationMethod(
            "genetic_drift",
            "non_learning",
            lambda scenario, environment: GeneticDriftController(
                evaluation_budget=genetic_budget,
                seed=scenario.spec.seed,
            ),
        ),
        EvaluationMethod(
            "random_feasible",
            "sanity_floor",
            lambda scenario, environment: RandomFeasibleController(
                seed=scenario.spec.seed, allow_hold=False
            ),
        ),
    )


def missing_required_methods(methods: tuple[EvaluationMethod, ...]) -> tuple[str, ...]:
    available = {method.name for method in methods}
    return tuple(name for name in REQUIRED_COMPARISON_METHODS if name not in available)


def evaluate_scenario(
    scenario: ScenarioInstance,
    method: EvaluationMethod,
) -> EvaluationRun:
    """Evaluate one method without mutating the reusable scenario instance."""

    environment = scenario.environment()
    controller = method.factory(scenario, environment)
    if controller.name != method.name:
        raise ValueError(
            f"controller name {controller.name!r} does not match method {method.name!r}"
        )
    if hasattr(controller, "begin_scenario"):
        controller.begin_scenario(scenario.spec.scenario_id)

    traces: list[SlotTrace] = []
    cumulative_completed = 0
    cumulative_misses = 0
    cumulative_reward = 0.0
    total_slots = scenario.spec.horizon_slots + scenario.spec.drain_slots
    for _ in range(total_slots):
        decision = controller.select_action(environment)
        result = environment.step(decision.action)
        cumulative_completed += len(result.completed_task_ids)
        cumulative_misses += len(result.new_deadline_miss_ids)
        cumulative_reward += result.lyapunov.reward

        if hasattr(controller, "observe_transition"):
            controller.observe_transition(result)
        adaptation_updates = 0
        adaptation_seconds = 0.0
        if (
            scenario.spec.change_slot is not None
            and result.slot == scenario.spec.change_slot
            and method.adaptation_updates > 0
            and hasattr(controller, "adapt")
        ):
            if hasattr(controller, "declare_change"):
                controller.declare_change(
                    f"{scenario.spec.scenario_id}:slot:{result.slot}", environment
                )
            start = perf_counter()
            adaptation_result = controller.adapt(method.adaptation_updates)
            adaptation_seconds = perf_counter() - start
            adaptation_updates = int(
                getattr(
                    adaptation_result,
                    "completed_updates",
                    adaptation_result if isinstance(adaptation_result, int) else 0,
                )
            )
        traces.append(
            make_slot_trace(
                result,
                offered_load_horizon=scenario.spec.horizon_slots,
                cumulative_completed=cumulative_completed,
                cumulative_misses=cumulative_misses,
                cumulative_reward=cumulative_reward,
                controller_seconds=decision.diagnostics.elapsed_seconds,
                candidates_evaluated=decision.diagnostics.candidates_evaluated,
                adaptation_updates=adaptation_updates,
                adaptation_seconds=adaptation_seconds,
            )
        )
        if result.terminated or result.truncated:
            break
        if (
            environment.slot >= scenario.spec.horizon_slots
            and len(environment.tasks) == len(scenario.arrivals)
            and all(record.is_complete for record in environment.tasks.values())
        ):
            break

    metrics, tasks = summarize_run(
        environment,
        traces,
        scheduled_tasks=len(scenario.arrivals),
        offered_load_slots=scenario.spec.horizon_slots,
        realized_offered_load_ratio=scenario.realized_offered_load_ratio,
        offline_training_seconds=method.offline_training_seconds,
        change_slot=scenario.spec.change_slot,
    )
    configuration = asdict(scenario.config)
    configuration_digest = sha256(
        json.dumps(configuration, default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()
    revision, dirty = code_identity()
    return EvaluationRun(
        schema_version="vec-evidence/v1",
        scenario_id=scenario.spec.scenario_id,
        split=scenario.spec.split,
        stress=scenario.spec.stress,
        seed=scenario.spec.seed,
        method=method.name,
        method_family=method.family,
        trace_digest=scenario.trace_digest,
        configuration_digest=configuration_digest,
        code_revision=revision,
        code_dirty=dirty,
        adaptation_updates_budget=method.adaptation_updates,
        checkpoint=method.checkpoint,
        metrics=metrics,
        traces=tuple(traces),
        tasks=tasks,
    )


def code_identity() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
