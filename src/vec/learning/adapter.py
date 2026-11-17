"""Tensor/action bridge between :class:`VecQueueEnv` and hybrid MADDPG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..control import AgentObservation, ControlObservation, DestinationMask
from ..env import (
    Destination,
    DispatchDecision,
    StepAction,
    StepResult,
    VecQueueEnv,
)
from ..safety import MotionCommand
from .networks import RoleSpec
from .replay import HybridTransition


_ROLE_FEATURES = {
    "vehicle": (
        "ingress_q",
        "local_compute_q",
        "x",
        "y",
        "z",
        "reachable_rsu_fraction",
        "reachable_uav_fraction",
    ),
    "rsu": (
        "upload_q",
        "receive_q",
        "compute_q",
        "cpu_capacity",
        "x",
        "y",
        "z",
    ),
    "uav": (
        "upload_q",
        "receive_q",
        "compute_q",
        "backhaul_q",
        "battery_fraction",
        "virtual_energy_q",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "active",
        "returning",
        "landed",
        "failed",
    ),
}


@dataclass(frozen=True)
class EncodedControlObservation:
    state: np.ndarray
    observations: Mapping[str, np.ndarray]
    task_features: Mapping[str, np.ndarray]
    task_masks: Mapping[str, np.ndarray]
    categorical_masks: Mapping[str, np.ndarray]
    task_ids: Mapping[str, tuple[tuple[int, ...], ...]]


class VecPolicyAdapter:
    """Use stable feature orders and pad variable task sets for the actors."""

    task_feature_dim = 7

    def __init__(self, environment: VecQueueEnv, max_tasks: int = 16) -> None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self.environment = environment
        self.max_tasks = max_tasks

    def role_specs(self) -> tuple[RoleSpec, ...]:
        config = self.environment.config
        specs: list[RoleSpec] = []
        counts = {
            "vehicle": len(config.vehicle_cpu_hz),
            "rsu": len(config.rsu_cpu_hz),
            "uav": len(config.uav_cpu_hz),
        }
        if counts["vehicle"]:
            specs.append(
                RoleSpec(
                    "vehicle",
                    counts["vehicle"],
                    len(_ROLE_FEATURES["vehicle"]),
                    self.task_feature_dim,
                    self.max_tasks,
                    categorical_dim=2 + counts["rsu"] + counts["uav"],
                    has_split=True,
                )
            )
        if counts["rsu"]:
            specs.append(
                RoleSpec(
                    "rsu",
                    counts["rsu"],
                    len(_ROLE_FEATURES["rsu"]),
                    self.task_feature_dim,
                    self.max_tasks,
                    has_cpu=True,
                )
            )
        if counts["uav"]:
            specs.append(
                RoleSpec(
                    "uav",
                    counts["uav"],
                    len(_ROLE_FEATURES["uav"]),
                    self.task_feature_dim,
                    self.max_tasks,
                    has_cpu=True,
                    has_motion=True,
                )
            )
        return tuple(specs)

    def encode(
        self,
        observation: ControlObservation | None = None,
        destination_masks: tuple[DestinationMask, ...] | None = None,
    ) -> EncodedControlObservation:
        observation = observation or self.environment.control_observation()
        destination_masks = (
            destination_masks or self.environment.destination_masks()
        )
        groups = {
            "vehicle": observation.vehicles,
            "rsu": observation.rsus,
            "uav": observation.uavs,
        }
        observations: dict[str, np.ndarray] = {}
        task_features: dict[str, np.ndarray] = {}
        task_masks: dict[str, np.ndarray] = {}
        categorical_masks: dict[str, np.ndarray] = {}
        task_ids: dict[str, tuple[tuple[int, ...], ...]] = {}
        specs = {spec.name: spec for spec in self.role_specs()}
        for role, agents in groups.items():
            if role not in specs:
                continue
            observations[role] = self._agent_features(role, agents)
            features, masks, ids = self._task_set(agents)
            task_features[role] = features
            task_masks[role] = masks
            task_ids[role] = ids
            if role == "vehicle":
                categorical_masks[role] = np.asarray(
                    [
                        (mask.hold, mask.local, *mask.rsu, *mask.uav)
                        for mask in destination_masks
                    ],
                    dtype=np.bool_,
                )
            else:
                categorical_masks[role] = np.zeros(
                    (len(agents), 0), dtype=np.bool_
                )
        return EncodedControlObservation(
            state=np.asarray(observation.global_critic_state, dtype=np.float32),
            observations=observations,
            task_features=task_features,
            task_masks=task_masks,
            categorical_masks=categorical_masks,
            task_ids=task_ids,
        )

    def decode(
        self,
        encoded: EncodedControlObservation,
        actions: Mapping[str, np.ndarray],
    ) -> StepAction:
        config = self.environment.config
        dispatches: list[DispatchDecision] = []
        cpu_weights: dict[str, dict[int, float]] = {}

        if "vehicle" in actions:
            destination_count = 2 + len(config.rsu_cpu_hz) + len(config.uav_cpu_hz)
            for vehicle_id, vector in enumerate(actions["vehicle"]):
                ids = encoded.task_ids["vehicle"][vehicle_id]
                if not ids:
                    continue
                category = int(np.argmax(vector[:destination_count]))
                if category == 0:
                    continue
                task_id = ids[0]
                fraction = float(np.clip(vector[destination_count], 0.0, 1.0))
                if category == 1:
                    destination = Destination.local(vehicle_id)
                    fraction = 1.0
                elif category < 2 + len(config.rsu_cpu_hz):
                    destination = Destination.rsu(category - 2)
                    fraction = 1.0
                else:
                    destination = Destination.uav(
                        category - 2 - len(config.rsu_cpu_hz)
                    )
                    if not self.environment.tasks[task_id].spec.partitionable:
                        fraction = 1.0
                dispatches.append(
                    DispatchDecision(
                        vehicle_id, task_id, destination, fraction
                    )
                )

        for role, prefix in (("rsu", "compute_rsu"), ("uav", "compute_uav")):
            if role not in actions:
                continue
            for agent_id, vector in enumerate(actions[role]):
                share_start = 1 + (3 if role == "uav" else 0)
                ids = encoded.task_ids[role][agent_id]
                weights = {
                    task_id: float(vector[share_start + index])
                    for index, task_id in enumerate(ids)
                    if vector[share_start + index] > 0.0
                }
                if weights:
                    cpu_weights[f"{prefix}:{agent_id}"] = weights

        motions: list[MotionCommand] = []
        frequencies: list[float] = []
        if "uav" in actions:
            limits = config.flight_limits
            horizontal = limits.max_speed_mps if limits is not None else 1.0
            climb = limits.max_climb_mps if limits is not None else 1.0
            descent = limits.max_descent_mps if limits is not None else 1.0
            for uav_id, vector in enumerate(actions["uav"]):
                frequencies.append(
                    float(np.clip(vector[0], 0.0, 1.0))
                    * config.uav_cpu_hz[uav_id]
                )
                vz = float(vector[3])
                motions.append(
                    MotionCommand(
                        (
                            float(vector[1]) * horizontal,
                            float(vector[2]) * horizontal,
                            vz * (climb if vz >= 0.0 else descent),
                        )
                    )
                )
        return StepAction(
            dispatches=tuple(dispatches),
            cpu_weights=cpu_weights,
            uav_motion_commands=tuple(motions),
            uav_cpu_frequencies_hz=tuple(frequencies),
        )

    def encode_executed_actions(
        self,
        encoded: EncodedControlObservation,
        proposed_actions: Mapping[str, np.ndarray],
        result: object,
    ) -> dict[str, np.ndarray]:
        """Reflect environment repair in replay's executed action vectors."""

        executed = {
            role: np.asarray(values, dtype=np.float32).copy()
            for role, values in proposed_actions.items()
        }
        trace = result.action_trace
        config = self.environment.config
        if "vehicle" in executed:
            width = 2 + len(config.rsu_cpu_hz) + len(config.uav_cpu_hz)
            executed["vehicle"][:, :width] = 0.0
            executed["vehicle"][:, 0] = 1.0
            by_vehicle = {item.vehicle_id: item for item in trace.dispatches}
            for vehicle_id, dispatch in by_vehicle.items():
                if dispatch.destination_kind == "local":
                    category = 1
                elif dispatch.destination_kind == "rsu":
                    category = 2 + dispatch.destination_id
                else:
                    category = 2 + len(config.rsu_cpu_hz) + dispatch.destination_id
                executed["vehicle"][vehicle_id, :width] = 0.0
                executed["vehicle"][vehicle_id, category] = 1.0
                executed["vehicle"][vehicle_id, width] = dispatch.uav_fraction
        if "uav" in executed:
            for uav_id, (frequency, velocity) in enumerate(
                zip(
                    trace.executed_uav_cpu_hz,
                    trace.executed_uav_velocities,
                )
            ):
                maximum = config.uav_cpu_hz[uav_id]
                executed["uav"][uav_id, 0] = (
                    frequency / maximum if maximum > 0.0 else 0.0
                )
                limits = config.flight_limits
                horizontal = limits.max_speed_mps if limits is not None else 1.0
                vertical = (
                    limits.max_climb_mps
                    if limits is not None and velocity[2] >= 0.0
                    else limits.max_descent_mps
                    if limits is not None
                    else 1.0
                )
                executed["uav"][uav_id, 1] = self._safe_ratio(
                    velocity[0], horizontal
                )
                executed["uav"][uav_id, 2] = self._safe_ratio(
                    velocity[1], horizontal
                )
                executed["uav"][uav_id, 3] = self._safe_ratio(
                    velocity[2], vertical
                )
        return executed

    def transition(
        self,
        before: EncodedControlObservation,
        after: EncodedControlObservation,
        proposed_actions: Mapping[str, np.ndarray],
        result: StepResult,
        rewards: Mapping[str, float] | None = None,
    ) -> HybridTransition:
        """Build a replay item from one real environment transition."""

        executed = self.encode_executed_actions(before, proposed_actions, result)
        role_rewards = rewards or {
            spec.name: result.lyapunov.reward for spec in self.role_specs()
        }
        return HybridTransition(
            state=before.state,
            next_state=after.state,
            observations=before.observations,
            next_observations=after.observations,
            task_features=before.task_features,
            next_task_features=after.task_features,
            task_masks=before.task_masks,
            next_task_masks=after.task_masks,
            categorical_masks=before.categorical_masks,
            next_categorical_masks=after.categorical_masks,
            proposed_actions={
                role: np.asarray(action, dtype=np.float32)
                for role, action in proposed_actions.items()
            },
            executed_actions=executed,
            rewards=role_rewards,
            done=result.terminated or result.truncated,
        )

    def _agent_features(
        self, role: str, agents: tuple[AgentObservation, ...]
    ) -> np.ndarray:
        names = _ROLE_FEATURES[role]
        return np.asarray(
            [[agent.features[name] for name in names] for agent in agents],
            dtype=np.float32,
        )

    def _task_set(
        self, agents: tuple[AgentObservation, ...]
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
        features = np.zeros(
            (len(agents), self.max_tasks, self.task_feature_dim),
            dtype=np.float32,
        )
        mask = np.zeros((len(agents), self.max_tasks), dtype=np.bool_)
        retained_ids: list[tuple[int, ...]] = []
        for agent_index, agent in enumerate(agents):
            ids = agent.eligible_task_ids[: self.max_tasks]
            retained_ids.append(ids)
            for task_index, task_id in enumerate(ids):
                features[agent_index, task_index] = self._task_features(task_id)
                mask[agent_index, task_index] = True
        return features, mask, tuple(retained_ids)

    def _task_features(self, task_id: int) -> np.ndarray:
        record = self.environment.tasks[task_id]
        spec = record.spec
        age = max(0.0, self.environment.time - spec.arrival_time)
        slack = spec.deadline - age
        qref = self.environment.config.lyapunov.queue_reference_cycles
        return np.asarray(
            (
                spec.input_bits / 1.0e6,
                spec.cpu_cycles / qref,
                spec.intensity / 1.0e3,
                age / spec.deadline,
                slack / spec.deadline,
                float(spec.partitionable),
                float(record.deadline_missed),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(np.clip(numerator / denominator, -1.0, 1.0))
