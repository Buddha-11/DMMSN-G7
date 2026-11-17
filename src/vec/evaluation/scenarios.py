"""Versioned, deterministic evaluation scenarios and split manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np

from ..energy import BatteryConfig
from ..env import ScheduledArrival, ScheduledHarvest, VecEnvConfig, VecQueueEnv
from ..mobility import Position3D, RecordedTraceMobility
from ..safety import FlightLimits


SCENARIO_SCHEMA_VERSION = "vec-evaluation/v1"
STANDARD_STRESSES = (
    "low_load",
    "near_capacity",
    "overload",
    "zero_harvest",
    "harvest_outage",
    "backhaul_bottleneck",
    "traffic_surge",
)
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: Split
    stress: str
    seed: int
    horizon_slots: int = 200
    drain_slots: int = 100
    nominal_load_ratio: float = 0.35
    surge_load_ratio: float | None = None
    change_slot: int | None = None
    mean_harvest_j: float = 140.0
    outage_start_slot: int | None = None
    outage_end_slot: int | None = None
    backhaul_bps: float = 20.0e6
    mobility_trace: str | None = None
    mobility_vehicle_ids: tuple[str, ...] = ()
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError("unsupported scenario schema version")
        if self.stress not in STANDARD_STRESSES:
            raise ValueError(f"unknown stress suite {self.stress!r}")
        if self.horizon_slots <= 0 or self.drain_slots < 0:
            raise ValueError("scenario horizons must be non-negative")
        if self.nominal_load_ratio < 0:
            raise ValueError("nominal_load_ratio must be non-negative")
        if self.surge_load_ratio is not None and self.surge_load_ratio < 0:
            raise ValueError("surge_load_ratio must be non-negative")
        if self.change_slot is not None and not 0 <= self.change_slot < self.horizon_slots:
            raise ValueError("change_slot must lie inside the offered-load horizon")
        object.__setattr__(
            self, "mobility_vehicle_ids", tuple(self.mobility_vehicle_ids)
        )


@dataclass(frozen=True)
class ScenarioManifest:
    manifest_id: str
    scenarios: tuple[ScenarioSpec, ...]
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario IDs must be unique")
        assignments: dict[int, set[str]] = {}
        mobility_assignments: dict[str, set[str]] = {}
        for scenario in self.scenarios:
            assignments.setdefault(scenario.seed, set()).add(scenario.split)
            if scenario.mobility_trace:
                mobility_assignments.setdefault(
                    scenario.mobility_trace, set()
                ).add(scenario.split)
        leaked = {seed: splits for seed, splits in assignments.items() if len(splits) > 1}
        if leaked:
            raise ValueError(f"seeds cross manifest splits: {leaked}")
        leaked_traces = {
            trace: splits
            for trace, splits in mobility_assignments.items()
            if len(splits) > 1
        }
        if leaked_traces:
            raise ValueError(
                f"mobility traces cross manifest splits: {leaked_traces}"
            )

    def select(
        self,
        *,
        split: Split | None = None,
        stresses: Iterable[str] | None = None,
    ) -> tuple[ScenarioSpec, ...]:
        selected_stresses = set(stresses) if stresses is not None else None
        return tuple(
            scenario
            for scenario in self.scenarios
            if (split is None or scenario.split == split)
            and (
                selected_stresses is None
                or scenario.stress in selected_stresses
            )
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCENARIO_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema version")
        return cls(
            manifest_id=payload["manifest_id"],
            scenarios=tuple(
                ScenarioSpec(**scenario) for scenario in payload["scenarios"]
            ),
            schema_version=payload["schema_version"],
        )


@dataclass(frozen=True)
class ScenarioInstance:
    spec: ScenarioSpec
    config: VecEnvConfig
    arrivals: tuple[ScheduledArrival, ...]
    harvests: tuple[ScheduledHarvest, ...]
    offered_cycles: float
    nominal_service_cycles_per_slot: float
    trace_digest: str
    mobility_frames: tuple[tuple[Position3D, ...], ...] = ()

    @property
    def realized_offered_load_ratio(self) -> float:
        denominator = (
            self.nominal_service_cycles_per_slot * self.spec.horizon_slots
        )
        return self.offered_cycles / denominator if denominator > 0 else float("inf")

    def environment(self) -> VecQueueEnv:
        mobility = (
            RecordedTraceMobility(self.mobility_frames, self.config.slot_duration)
            if self.mobility_frames
            else None
        )
        return VecQueueEnv(
            self.config, self.arrivals, self.harvests, mobility_provider=mobility
        )


def create_standard_manifest(
    seeds_by_split: Mapping[Split, Iterable[int]] | None = None,
    *,
    horizon_slots: int = 200,
    drain_slots: int = 100,
    manifest_id: str = "standard-synthetic-v1",
) -> ScenarioManifest:
    """Create complete-scenario splits; no trace is sliced across splits."""

    seeds_by_split = seeds_by_split or {
        "train": range(0, 10),
        "validation": range(100, 105),
        "test": range(200, 210),
    }
    scenarios: list[ScenarioSpec] = []
    for split, seeds in seeds_by_split.items():
        for seed in seeds:
            for stress in STANDARD_STRESSES:
                values: dict = {
                    "nominal_load_ratio": 0.35,
                    "mean_harvest_j": 140.0,
                    "backhaul_bps": 20.0e6,
                }
                if stress == "near_capacity":
                    values["nominal_load_ratio"] = 0.85
                elif stress == "overload":
                    values["nominal_load_ratio"] = 1.25
                elif stress == "zero_harvest":
                    values["mean_harvest_j"] = 0.0
                elif stress == "harvest_outage":
                    values["outage_start_slot"] = horizon_slots // 3
                    values["outage_end_slot"] = 2 * horizon_slots // 3
                    values["change_slot"] = horizon_slots // 3
                elif stress == "backhaul_bottleneck":
                    values["backhaul_bps"] = 0.5e6
                elif stress == "traffic_surge":
                    values["change_slot"] = horizon_slots // 2
                    values["surge_load_ratio"] = 1.25
                scenario_id = f"{split}-{stress}-s{int(seed):04d}"
                scenarios.append(
                    ScenarioSpec(
                        scenario_id=scenario_id,
                        split=split,
                        stress=stress,
                        seed=int(seed),
                        horizon_slots=horizon_slots,
                        drain_slots=drain_slots,
                        **values,
                    )
                )
    return ScenarioManifest(manifest_id, tuple(scenarios))


def instantiate_scenario(spec: ScenarioSpec) -> ScenarioInstance:
    """Materialize traffic and harvesting once for paired algorithm runs."""

    random = np.random.default_rng(spec.seed)
    vehicle_cpu = (0.5e9, 0.5e9, 0.5e9)
    rsu_cpu = (3.0e9, 3.0e9)
    uav_cpu = (2.0e9, 2.0e9)
    nominal_service = sum(vehicle_cpu) + sum(rsu_cpu) + sum(uav_cpu)
    config = VecEnvConfig(
        vehicle_cpu_hz=vehicle_cpu,
        rsu_cpu_hz=rsu_cpu,
        uav_cpu_hz=uav_cpu,
        rsu_uplink_bps=(20.0e6, 20.0e6),
        uav_uplink_bps=(12.0e6, 12.0e6),
        uav_backhaul_bps=spec.backhaul_bps,
        cloud_cpu_hz=12.0e9,
        vehicle_static_positions=(
            (100.0, 100.0, 0.0),
            (450.0, 150.0, 0.0),
            (800.0, 100.0, 0.0),
        ),
        rsu_positions=((250.0, 100.0, 0.0), (650.0, 100.0, 0.0)),
        uav_positions=((300.0, 250.0, 100.0), (700.0, 250.0, 100.0)),
        # Each UAV has its own vertically aligned landing site.  Declaring the
        # sites activates the verified return planner used by the battery
        # reserve constraint; without them a zero-harvest research episode can
        # only hover until exhaustion and is incorrectly guaranteed to fail.
        uav_landing_positions=((300.0, 250.0, 50.0), (700.0, 250.0, 50.0)),
        uav_batteries=(
            BatteryConfig(50_000.0, 45_000.0),
            BatteryConfig(50_000.0, 45_000.0),
        ),
        flight_limits=FlightLimits(
            x_min_m=0.0,
            x_max_m=1_000.0,
            y_min_m=0.0,
            y_max_m=1_000.0,
            altitude_min_m=50.0,
            altitude_max_m=150.0,
            max_speed_mps=20.0,
            max_climb_mps=5.0,
            max_descent_mps=5.0,
            max_acceleration_mps2=10.0,
            minimum_separation_m=50.0,
        ),
        max_episode_slots=spec.horizon_slots + spec.drain_slots,
    )

    mean_cycles = 4.0e9
    arrivals: list[ScheduledArrival] = []
    task_id = 0
    offered_cycles = 0.0
    for slot in range(spec.horizon_slots):
        ratio = (
            spec.surge_load_ratio
            if spec.change_slot is not None
            and slot >= spec.change_slot
            and spec.surge_load_ratio is not None
            else spec.nominal_load_ratio
        )
        task_rate = ratio * nominal_service / mean_cycles
        for _ in range(int(random.poisson(task_rate))):
            vehicle_id = int(random.integers(0, len(vehicle_cpu)))
            input_bits = float(random.uniform(2.0e6, 6.0e6))
            cycles = float(mean_cycles * random.uniform(0.8, 1.2))
            deadline = float(random.uniform(8.0, 20.0))
            arrivals.append(
                ScheduledArrival(
                    slot=slot,
                    vehicle_id=vehicle_id,
                    input_bits=input_bits,
                    cpu_cycles=cycles,
                    deadline=deadline,
                    task_id=task_id,
                    partitionable=bool(random.random() < 0.5),
                )
            )
            offered_cycles += cycles
            task_id += 1

    harvests: list[ScheduledHarvest] = []
    for slot in range(spec.horizon_slots + spec.drain_slots):
        in_outage = (
            spec.outage_start_slot is not None
            and spec.outage_end_slot is not None
            and spec.outage_start_slot <= slot < spec.outage_end_slot
        )
        for uav_id in range(len(uav_cpu)):
            harvested = 0.0 if in_outage else spec.mean_harvest_j
            if harvested > 0.0:
                # Bounded nonnegative variation, shared by every paired method.
                harvested *= float(random.uniform(0.8, 1.2))
                harvests.append(ScheduledHarvest(slot, uav_id, harvested))

    mobility_frames = (
        load_recorded_mobility_csv(
            spec.mobility_trace, spec.mobility_vehicle_ids
        )
        if spec.mobility_trace
        else ()
    )
    if mobility_frames and len(mobility_frames[0]) != len(vehicle_cpu):
        raise ValueError(
            "the standard topology requires exactly three recorded vehicles"
        )
    digest_payload = {
        "spec": asdict(spec),
        "arrivals": [asdict(item) for item in arrivals],
        "harvests": [asdict(item) for item in harvests],
        "mobility_frames": mobility_frames,
    }
    digest = sha256(
        json.dumps(digest_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ScenarioInstance(
        spec=spec,
        config=config,
        arrivals=tuple(arrivals),
        harvests=tuple(harvests),
        offered_cycles=offered_cycles,
        nominal_service_cycles_per_slot=nominal_service,
        trace_digest=digest,
        mobility_frames=mobility_frames,
    )


def load_recorded_mobility_csv(
    path: str | Path,
    vehicle_ids: Iterable[str] = (),
) -> tuple[tuple[Position3D, ...], ...]:
    """Load complete fixed-population frames exported from SUMO/libsumo.

    Required columns are ``slot,vehicle_id,x,y``; ``z`` is optional. Every
    selected vehicle must occur in every frame, preventing silent interpolation
    across missing observations.
    """

    source = Path(path)
    by_slot: dict[int, dict[str, Position3D]] = {}
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"slot", "vehicle_id", "x", "y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"mobility CSV requires columns {sorted(required)}")
        for row in reader:
            slot = int(row["slot"])
            if slot < 0:
                raise ValueError("mobility slots must be non-negative")
            vehicle_id = row["vehicle_id"]
            position = (
                float(row["x"]),
                float(row["y"]),
                float(row.get("z") or 0.0),
            )
            if vehicle_id in by_slot.setdefault(slot, {}):
                raise ValueError(
                    f"duplicate mobility row for slot={slot}, vehicle={vehicle_id}"
                )
            by_slot[slot][vehicle_id] = position
    if not by_slot:
        raise ValueError("mobility CSV is empty")
    expected_slots = list(range(max(by_slot) + 1))
    if sorted(by_slot) != expected_slots:
        raise ValueError("mobility slots must be contiguous and start at zero")
    selected = tuple(vehicle_ids) or tuple(sorted(by_slot[0]))
    if not selected:
        raise ValueError("mobility trace has no vehicles")
    frames: list[tuple[Position3D, ...]] = []
    for slot in expected_slots:
        missing = [vehicle_id for vehicle_id in selected if vehicle_id not in by_slot[slot]]
        if missing:
            raise ValueError(f"slot {slot} misses selected vehicles {missing}")
        frames.append(tuple(by_slot[slot][vehicle_id] for vehicle_id in selected))
    return tuple(frames)
