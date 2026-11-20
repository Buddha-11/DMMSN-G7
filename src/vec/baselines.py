"""Fair non-learning controllers sharing the validated VEC execution path."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt
from random import Random
from time import perf_counter
from typing import Protocol, Sequence

from .energy import SlotEnergyDemand
from .env import (
    Destination,
    DispatchDecision,
    StepAction,
    StepResult,
    UAVMode,
    VecQueueEnv,
)
from .lyapunov import LyapunovSlotInput
from .queues import QueueKey, QueueStage
from .tasks import DestinationKind, TaskRecord


@dataclass(frozen=True)
class BaselineDiagnostics:
    controller: str
    candidates_evaluated: int
    evaluation_budget: int | None
    elapsed_seconds: float
    best_score: float | None = None
    generations: int = 0
    seed: int | None = None


@dataclass(frozen=True)
class BaselineDecision:
    action: StepAction
    diagnostics: BaselineDiagnostics


@dataclass(frozen=True)
class BaselineStep:
    decision: BaselineDecision
    result: StepResult


@dataclass(frozen=True)
class BaselineRolloutMetrics:
    controller: str
    slots: int
    completed_tasks: int
    deadline_misses: int
    total_energy_j: float
    cumulative_reward: float
    peak_backlog_cycles: float
    candidates_evaluated: int
    controller_seconds: float


@dataclass(frozen=True)
class BaselineRollout:
    steps: tuple[BaselineStep, ...]
    metrics: BaselineRolloutMetrics


class BaselineController(Protocol):
    name: str

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        ...


@dataclass(frozen=True)
class RoutingOption:
    vehicle_id: int
    task_id: int | None
    destination: Destination | None
    uav_fraction: float = 1.0

    @property
    def is_hold(self) -> bool:
        return self.destination is None or self.task_id is None

    @property
    def sort_key(self) -> tuple[int, int, int, float]:
        if self.is_hold:
            return (0, self.vehicle_id, -1, 1.0)
        order = {
            DestinationKind.LOCAL: 1,
            DestinationKind.RSU: 2,
            DestinationKind.UAV: 3,
        }
        return (
            order[self.destination.kind],
            self.vehicle_id,
            self.destination.node_id,
            self.uav_fraction,
        )

    def dispatch(self) -> DispatchDecision | None:
        if self.is_hold:
            return None
        return DispatchDecision(
            self.vehicle_id,
            self.task_id,
            self.destination,
            self.uav_fraction,
        )


def execute_controller_step(
    env: VecQueueEnv, controller: BaselineController
) -> BaselineStep:
    """The only baseline execution path: proposal then normal environment step."""

    decision = controller.select_action(env)
    return BaselineStep(decision=decision, result=env.step(decision.action))


def run_controller_episode(
    env: VecQueueEnv,
    controller: BaselineController,
    max_steps: int,
) -> BaselineRollout:
    """Run any baseline through identical transition and metric collection."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    steps: list[BaselineStep] = []
    for _ in range(max_steps):
        step = execute_controller_step(env, controller)
        steps.append(step)
        if step.result.terminated or step.result.truncated:
            break
    return BaselineRollout(
        steps=tuple(steps),
        metrics=BaselineRolloutMetrics(
            controller=controller.name,
            slots=len(steps),
            completed_tasks=sum(
                len(step.result.completed_task_ids) for step in steps
            ),
            deadline_misses=sum(
                len(step.result.new_deadline_miss_ids) for step in steps
            ),
            total_energy_j=sum(step.result.energy.total_j for step in steps),
            cumulative_reward=sum(step.result.lyapunov.reward for step in steps),
            peak_backlog_cycles=max(
                (sum(step.result.queue_after.values()) for step in steps),
                default=0.0,
            ),
            candidates_evaluated=sum(
                step.decision.diagnostics.candidates_evaluated for step in steps
            ),
            controller_seconds=sum(
                step.decision.diagnostics.elapsed_seconds for step in steps
            ),
        ),
    )


class LocalOnlyController:
    name = "local_only"

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        start = perf_counter()
        decisions = []
        for mask in env.destination_masks():
            if mask.ingress_task_ids:
                decisions.append(
                    DispatchDecision(
                        mask.vehicle_id,
                        mask.ingress_task_ids[0],
                        Destination.local(mask.vehicle_id),
                    )
                )
        return _decision(self.name, decisions, len(decisions), start)


class NearestFeasibleEdgeController:
    name = "nearest_feasible_edge"

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        start = perf_counter()
        decisions = []
        evaluated = 0
        for mask in env.destination_masks():
            if not mask.ingress_task_ids:
                continue
            vehicle = env.vehicle_positions[mask.vehicle_id]
            candidates: list[tuple[float, int, int, Destination]] = []
            for rsu_id, feasible in enumerate(mask.rsu):
                if feasible:
                    candidates.append(
                        (
                            _distance(vehicle, env.rsu_positions[rsu_id]),
                            0,
                            rsu_id,
                            Destination.rsu(rsu_id),
                        )
                    )
            for uav_id, feasible in enumerate(mask.uav):
                if feasible:
                    candidates.append(
                        (
                            _distance(vehicle, env.uav_positions[uav_id]),
                            1,
                            uav_id,
                            Destination.uav(uav_id),
                        )
                    )
            evaluated += len(candidates)
            destination = (
                min(candidates, key=lambda item: item[:3])[3]
                if candidates
                else Destination.local(mask.vehicle_id)
            )
            decisions.append(
                DispatchDecision(
                    mask.vehicle_id,
                    mask.ingress_task_ids[0],
                    destination,
                )
            )
        return _decision(self.name, decisions, evaluated, start)


@dataclass(frozen=True)
class GreedyConfig:
    latency_weight: float = 1.0
    energy_weight: float = 0.05
    battery_scarcity_weight: float = 1.0
    split_grid: tuple[float, ...] = (0.0, 0.5, 1.0)

    def __post_init__(self) -> None:
        if min(
            self.latency_weight,
            self.energy_weight,
            self.battery_scarcity_weight,
        ) < 0:
            raise ValueError("greedy weights must be non-negative")
        if not self.split_grid or any(
            not 0.0 <= value <= 1.0 for value in self.split_grid
        ):
            raise ValueError("split_grid values must lie in [0, 1]")


class QueueEnergyGreedyController:
    name = "queue_energy_greedy"

    def __init__(self, config: GreedyConfig | None = None) -> None:
        self.config = config or GreedyConfig()

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        start = perf_counter()
        selected: list[RoutingOption] = []
        evaluated = 0
        for options in _routing_options(env, self.config.split_grid):
            service_options = [option for option in options if not option.is_hold]
            if not service_options:
                continue
            scored = [
                (self.score_option(env, option), option.sort_key, option)
                for option in service_options
            ]
            evaluated += len(scored)
            selected.append(min(scored, key=lambda item: (item[0], item[1]))[2])
        return _options_decision(self.name, selected, evaluated, start)

    def score_option(self, env: VecQueueEnv, option: RoutingOption) -> float:
        if option.is_hold:
            return inf
        record = env.tasks[option.task_id]
        destination = option.destination
        latency, energy, scarcity = _estimate_route(
            env, record, destination, option.uav_fraction
        )
        return (
            self.config.latency_weight * latency
            + self.config.energy_weight * energy
            + self.config.battery_scarcity_weight * scarcity
        )


class RandomFeasibleController:
    name = "random_feasible"

    def __init__(self, seed: int = 0, *, allow_hold: bool = True) -> None:
        self.seed = seed
        self.allow_hold = allow_hold
        self._random = Random(seed)

    def reset(self) -> None:
        self._random.seed(self.seed)

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        start = perf_counter()
        selected = []
        evaluated = 0
        for options in _routing_options(env, (0.0, 0.5, 1.0)):
            choices = list(options if self.allow_hold else options[1:])
            if not choices:
                continue
            evaluated += len(choices)
            selected.append(choices[self._random.randrange(len(choices))])
        return _options_decision(
            self.name, selected, evaluated, start, seed=self.seed
        )


class DriftCandidateEvaluator:
    """Deterministic one-slot surrogate using the common Lyapunov tracker."""

    def evaluate(
        self, env: VecQueueEnv, options: Sequence[RoutingOption]
    ) -> float:
        before = env.queue_backlogs()
        arrivals: dict[str, float] = {}
        departures: dict[str, float] = {}
        for option in options:
            if option.is_hold:
                continue
            record = env.tasks[option.task_id]
            amount = record.spec.cpu_cycles
            source = QueueKey(QueueStage.INGRESS, option.vehicle_id).label
            target, _ = env._dispatch_target(option.destination)
            departures[source] = departures.get(source, 0.0) + amount
            arrivals[target.label] = arrivals.get(target.label, 0.0) + amount

        compute_service = 0.0
        system_energy = 0.0
        uav_energy = [0.0 for _ in env.config.uav_cpu_hz]
        compute_specs: list[tuple[QueueKey, float, float, int | None]] = []
        compute_specs.extend(
            (
                QueueKey(QueueStage.COMPUTE_LOCAL, node_id),
                frequency,
                env.config.system_energy.vehicle_compute_kappa,
                None,
            )
            for node_id, frequency in enumerate(env.config.vehicle_cpu_hz)
        )
        compute_specs.extend(
            (
                QueueKey(QueueStage.COMPUTE_RSU, node_id),
                frequency,
                env.config.system_energy.rsu_compute_kappa,
                None,
            )
            for node_id, frequency in enumerate(env.config.rsu_cpu_hz)
        )
        compute_specs.extend(
            (
                QueueKey(QueueStage.COMPUTE_UAV, node_id),
                frequency,
                env.config.uav_energy.compute_kappa,
                node_id,
            )
            for node_id, frequency in enumerate(env.config.uav_cpu_hz)
        )
        compute_specs.append(
            (
                QueueKey(QueueStage.COMPUTE_CLOUD, 0),
                env.config.cloud_cpu_hz,
                env.config.system_energy.cloud_compute_kappa,
                None,
            )
        )
        for key, frequency, kappa, uav_id in compute_specs:
            available = before.get(key.label, 0.0) + arrivals.get(key.label, 0.0)
            served = min(available, frequency * env.config.slot_duration)
            if served <= 0:
                continue
            departures[key.label] = departures.get(key.label, 0.0) + served
            compute_service += served
            energy = kappa * frequency**2 * served
            if uav_id is None:
                system_energy += energy
            else:
                uav_energy[uav_id] += energy

        for uav_id, mode in enumerate(env.uav_modes):
            baseline = env.energy_model.slot_energy(
                SlotEnergyDemand(
                    control_active_seconds=(
                        0.0 if mode is UAVMode.LANDED else env.config.slot_duration
                    ),
                    airborne=mode is not UAVMode.LANDED,
                ),
                env.config.slot_duration,
            ).total_j
            uav_energy[uav_id] += baseline
        system_energy += sum(uav_energy)

        after = {
            label: max(
                backlog
                + arrivals.get(label, 0.0)
                - departures.get(label, 0.0),
                0.0,
            )
            for label, backlog in before.items()
        }
        misses = sum(
            not record.is_complete
            and not record.deadline_missed
            and env.time + env.config.slot_duration - record.spec.arrival_time
            > record.spec.deadline
            for record in env.tasks.values()
        )
        preview = env.lyapunov_tracker.preview(
            LyapunovSlotInput(
                queue_before=before,
                queue_after=after,
                queue_arrivals=arrivals,
                queue_departures=departures,
                uav_energy_j=tuple(uav_energy),
                harvest_j=tuple(0.0 for _ in uav_energy),
                system_energy_j=system_energy,
                unfinished_tasks=sum(
                    not record.is_complete for record in env.tasks.values()
                ),
                new_deadline_misses=misses,
                slot_duration_s=env.config.slot_duration,
                external_arrival_cycles=0.0,
                compute_service_cycles=compute_service,
            )
        )
        route_tiebreak = sum(
            _estimate_route(
                env,
                env.tasks[option.task_id],
                option.destination,
                option.uav_fraction,
            )[0]
            for option in options
            if not option.is_hold
        )
        hold_count = sum(option.is_hold for option in options)
        return preview.drift_plus_penalty + 1.0e-6 * route_tiebreak + hold_count


class GeneticDriftController:
    name = "genetic_drift"

    def __init__(
        self,
        evaluation_budget: int = 64,
        *,
        population_size: int = 8,
        mutation_probability: float = 0.15,
        seed: int = 0,
        evaluator: DriftCandidateEvaluator | None = None,
    ) -> None:
        if evaluation_budget <= 0 or population_size <= 0:
            raise ValueError("genetic budgets must be positive")
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1]")
        self.evaluation_budget = evaluation_budget
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.seed = seed
        self.evaluator = evaluator or DriftCandidateEvaluator()

    def select_action(self, env: VecQueueEnv) -> BaselineDecision:
        start = perf_counter()
        option_sets = _routing_options(env, (0.0, 0.5, 1.0))
        random = Random(self.seed + env.slot)
        local_genome = tuple(
            next(
                (
                    index
                    for index, option in enumerate(options)
                    if option.destination is not None
                    and option.destination.kind is DestinationKind.LOCAL
                ),
                0,
            )
            for options in option_sets
        )
        greedy = QueueEnergyGreedyController()
        greedy_genome = tuple(
            min(
                range(len(options)),
                key=lambda index: (
                    greedy.score_option(env, options[index]),
                    options[index].sort_key,
                ),
            )
            if len(options) > 1
            else 0
            for options in option_sets
        )
        population = [local_genome, greedy_genome]
        while len(population) < self.population_size:
            population.append(
                tuple(random.randrange(len(options)) for options in option_sets)
            )

        evaluated = 0
        generations = 0
        best_score = inf
        best_genome = local_genome
        ranked: list[tuple[float, tuple[int, ...]]] = []
        while evaluated < self.evaluation_budget:
            ranked = []
            for genome in population:
                if evaluated >= self.evaluation_budget:
                    break
                selected = tuple(
                    options[index]
                    for options, index in zip(option_sets, genome)
                )
                score = self.evaluator.evaluate(env, selected)
                evaluated += 1
                ranked.append((score, genome))
                if (score, genome) < (best_score, best_genome):
                    best_score, best_genome = score, genome
            generations += 1
            ranked.sort()
            parents = [genome for _, genome in ranked[: max(1, len(ranked) // 2)]]
            population = list(parents)
            while len(population) < self.population_size:
                first = parents[random.randrange(len(parents))]
                second = parents[random.randrange(len(parents))]
                child = []
                for gene_id, options in enumerate(option_sets):
                    gene = first[gene_id] if random.random() < 0.5 else second[gene_id]
                    if random.random() < self.mutation_probability:
                        gene = random.randrange(len(options))
                    child.append(gene)
                population.append(tuple(child))

        selected = tuple(
            options[index] for options, index in zip(option_sets, best_genome)
        )
        decisions = tuple(
            dispatch for option in selected if (dispatch := option.dispatch()) is not None
        )
        return BaselineDecision(
            action=StepAction(
                dispatches=decisions,
                uav_cpu_frequencies_hz=env.config.uav_cpu_hz,
            ),
            diagnostics=BaselineDiagnostics(
                controller=self.name,
                candidates_evaluated=evaluated,
                evaluation_budget=self.evaluation_budget,
                elapsed_seconds=perf_counter() - start,
                best_score=best_score,
                generations=generations,
                seed=self.seed,
            ),
        )


def _routing_options(
    env: VecQueueEnv, split_grid: Sequence[float]
) -> tuple[tuple[RoutingOption, ...], ...]:
    all_options = []
    for mask in env.destination_masks():
        options = [RoutingOption(mask.vehicle_id, None, None)]
        if not mask.ingress_task_ids:
            all_options.append(tuple(options))
            continue
        task_id = mask.ingress_task_ids[0]
        record = env.tasks[task_id]
        options.append(
            RoutingOption(
                mask.vehicle_id,
                task_id,
                Destination.local(mask.vehicle_id),
            )
        )
        options.extend(
            RoutingOption(mask.vehicle_id, task_id, Destination.rsu(rsu_id))
            for rsu_id, feasible in enumerate(mask.rsu)
            if feasible
        )
        for uav_id, feasible in enumerate(mask.uav):
            if not feasible:
                continue
            fractions = split_grid if record.spec.partitionable else (1.0,)
            options.extend(
                RoutingOption(
                    mask.vehicle_id,
                    task_id,
                    Destination.uav(uav_id),
                    float(fraction),
                )
                for fraction in fractions
            )
        all_options.append(tuple(sorted(options, key=lambda option: option.sort_key)))
    return tuple(all_options)


def _estimate_route(
    env: VecQueueEnv,
    record: TaskRecord,
    destination: Destination,
    uav_fraction: float,
) -> tuple[float, float, float]:
    spec = record.spec
    cfg = env.config
    if destination.kind is DestinationKind.LOCAL:
        frequency = cfg.vehicle_cpu_hz[destination.node_id]
        queue = env.get_queue(
            QueueStage.COMPUTE_LOCAL, destination.node_id
        ).equivalent_backlog
        latency = (queue + spec.cpu_cycles) / frequency if frequency > 0 else inf
        energy = (
            cfg.system_energy.vehicle_compute_kappa
            * frequency**2
            * spec.cpu_cycles
        )
        return latency, energy, 0.0

    rate = _uplink_rate(env, record.spec.vehicle_id, destination)
    upload = spec.input_bits / rate if rate > 0 else inf
    vehicle_energy = env._vehicle_tx_power_w() * upload
    if destination.kind is DestinationKind.RSU:
        frequency = cfg.rsu_cpu_hz[destination.node_id]
        queue = env.get_queue(
            QueueStage.COMPUTE_RSU, destination.node_id
        ).equivalent_backlog
        compute = (queue + spec.cpu_cycles) / frequency if frequency > 0 else inf
        energy = vehicle_energy + (
            cfg.system_energy.rsu_compute_kappa
            * frequency**2
            * spec.cpu_cycles
            + cfg.system_energy.rsu_receive_power_w * upload
        )
        return upload + compute, energy, 0.0

    uav_id = destination.node_id
    uav_frequency = cfg.uav_cpu_hz[uav_id]
    uav_queue = env.get_queue(
        QueueStage.COMPUTE_UAV, uav_id
    ).equivalent_backlog
    uav_cycles = uav_fraction * spec.cpu_cycles
    uav_compute = (
        (uav_queue + uav_cycles) / uav_frequency
        if uav_cycles > 0 and uav_frequency > 0
        else (0.0 if uav_cycles == 0 else inf)
    )
    cloud_cycles = (1.0 - uav_fraction) * spec.cpu_cycles
    cloud_bits = (1.0 - uav_fraction) * spec.input_bits
    backhaul_queue = env.get_queue(
        QueueStage.BACKHAUL_UAV, uav_id
    ).native_backlog
    backhaul = (
        (backhaul_queue + cloud_bits) / cfg.uav_backhaul_bps
        if cloud_bits > 0 and cfg.uav_backhaul_bps > 0
        else (0.0 if cloud_bits == 0 else inf)
    )
    cloud_queue = env.get_queue(
        QueueStage.COMPUTE_CLOUD, 0
    ).equivalent_backlog
    cloud_compute = (
        (cloud_queue + cloud_cycles) / cfg.cloud_cpu_hz
        if cloud_cycles > 0 and cfg.cloud_cpu_hz > 0
        else (0.0 if cloud_cycles == 0 else inf)
    )
    branch_latency = max(uav_compute, backhaul + cloud_compute)
    energy = vehicle_energy + cfg.uav_energy.receive_power_w * upload
    energy += cfg.uav_energy.compute_kappa * uav_frequency**2 * uav_cycles
    energy += cfg.uav_energy.backhaul_tx_power_w * backhaul
    if env.batteries:
        fraction = (
            env.batteries[uav_id].charge_j
            / env.batteries[uav_id].config.capacity_j
        )
        scarcity = 1.0 - fraction
    else:
        scarcity = 0.0
    return upload + branch_latency + cfg.merge_delay_s, energy, scarcity


def _uplink_rate(
    env: VecQueueEnv, vehicle_id: int, destination: Destination
) -> float:
    if env.config.radio is None:
        return (
            env.config.rsu_uplink_bps[destination.node_id]
            if destination.kind is DestinationKind.RSU
            else env.config.uav_uplink_bps[destination.node_id]
        )
    vehicle = env.vehicle_positions[vehicle_id]
    if destination.kind is DestinationKind.RSU:
        node = env.rsu_positions[destination.node_id]
        gain = env.radio_scheduler.channel.terrestrial_gain(vehicle, node)
        bandwidth = env.config.rsu_bandwidth_hz[destination.node_id]
    else:
        node = env.uav_positions[destination.node_id]
        gain = env.radio_scheduler.channel.aerial_gain(vehicle, node)
        bandwidth = env.config.uav_bandwidth_hz[destination.node_id]
    return env.radio_scheduler.channel.rate(gain, bandwidth)


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _decision(
    name: str,
    dispatches: Sequence[DispatchDecision],
    evaluated: int,
    start: float,
    *,
    seed: int | None = None,
) -> BaselineDecision:
    return BaselineDecision(
        action=StepAction(dispatches=tuple(dispatches)),
        diagnostics=BaselineDiagnostics(
            controller=name,
            candidates_evaluated=evaluated,
            evaluation_budget=None,
            elapsed_seconds=perf_counter() - start,
            seed=seed,
        ),
    )


def _options_decision(
    name: str,
    options: Sequence[RoutingOption],
    evaluated: int,
    start: float,
    *,
    seed: int | None = None,
) -> BaselineDecision:
    dispatches = tuple(
        dispatch for option in options if (dispatch := option.dispatch()) is not None
    )
    return _decision(name, dispatches, evaluated, start, seed=seed)
