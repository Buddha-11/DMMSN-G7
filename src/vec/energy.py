"""UAV propulsion, service-energy, harvesting, and battery accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class RotaryWingEnergyConfig:
    """Calibrated rotary-wing and electronics energy parameters."""

    profile_power_w: float = 79.86
    induced_power_w: float = 88.63
    rotor_tip_speed_mps: float = 120.0
    induced_velocity_mps: float = 4.03
    profile_speed_factor: float = 3.0
    parasitic_coefficient: float = 0.0093
    ascent_power_per_mps_w: float = 30.0
    descent_power_per_mps_w: float = 15.0
    vertical_quadratic_w_per_mps2: float = 2.0
    compute_kappa: float = 1.0e-28
    receive_power_w: float = 2.0
    backhaul_tx_power_w: float = 5.0
    control_power_w: float = 1.0

    def __post_init__(self) -> None:
        values = vars(self).values()
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("energy parameters must be finite and non-negative")
        if self.rotor_tip_speed_mps <= 0 or self.induced_velocity_mps <= 0:
            raise ValueError("rotor speeds must be positive")


@dataclass(frozen=True)
class SlotEnergyDemand:
    horizontal_speed_mps: float = 0.0
    vertical_speed_mps: float = 0.0
    cpu_frequency_hz: float = 0.0
    computed_cycles: float = 0.0
    receive_active_seconds: float = 0.0
    backhaul_active_seconds: float = 0.0
    control_active_seconds: float = 0.0
    airborne: bool = True


@dataclass(frozen=True)
class EnergyBreakdown:
    propulsion_j: float
    computation_j: float
    reception_j: float
    backhaul_j: float
    control_j: float

    @property
    def total_j(self) -> float:
        return (
            self.propulsion_j
            + self.computation_j
            + self.reception_j
            + self.backhaul_j
            + self.control_j
        )


class UAVEnergyModel:
    def __init__(self, config: RotaryWingEnergyConfig | None = None) -> None:
        self.config = config or RotaryWingEnergyConfig()

    def propulsion_power(
        self, horizontal_speed_mps: float, vertical_speed_mps: float, *, airborne: bool
    ) -> float:
        if not airborne:
            return 0.0
        horizontal = abs(float(horizontal_speed_mps))
        vertical = float(vertical_speed_mps)
        cfg = self.config
        profile = cfg.profile_power_w * (
            1.0
            + cfg.profile_speed_factor
            * horizontal**2
            / cfg.rotor_tip_speed_mps**2
        )
        induced_inner = sqrt(
            1.0 + horizontal**4 / (4.0 * cfg.induced_velocity_mps**4)
        ) - horizontal**2 / (2.0 * cfg.induced_velocity_mps**2)
        induced = cfg.induced_power_w * sqrt(max(induced_inner, 0.0))
        parasitic = cfg.parasitic_coefficient * horizontal**3
        vertical_linear = (
            cfg.ascent_power_per_mps_w * vertical
            if vertical >= 0
            else cfg.descent_power_per_mps_w * abs(vertical)
        )
        vertical_quadratic = cfg.vertical_quadratic_w_per_mps2 * vertical**2
        return max(
            profile + induced + parasitic + vertical_linear + vertical_quadratic,
            0.0,
        )

    def slot_energy(self, demand: SlotEnergyDemand, slot_duration: float) -> EnergyBreakdown:
        if not isfinite(slot_duration) or slot_duration <= 0:
            raise ValueError("slot_duration must be finite and positive")
        if not isfinite(demand.vertical_speed_mps):
            raise ValueError("vertical_speed_mps must be finite")
        for name, value in vars(demand).items():
            if name in ("vertical_speed_mps", "airborne"):
                continue
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        propulsion = self.propulsion_power(
            demand.horizontal_speed_mps,
            demand.vertical_speed_mps,
            airborne=demand.airborne,
        ) * slot_duration
        computation = (
            self.config.compute_kappa
            * demand.cpu_frequency_hz**2
            * demand.computed_cycles
        )
        return EnergyBreakdown(
            propulsion_j=propulsion,
            computation_j=computation,
            reception_j=self.config.receive_power_w * demand.receive_active_seconds,
            backhaul_j=(
                self.config.backhaul_tx_power_w * demand.backhaul_active_seconds
            ),
            control_j=self.config.control_power_w * demand.control_active_seconds,
        )


@dataclass(frozen=True)
class SystemEnergyConfig:
    """Energy parameters for non-UAV compute and communication owners."""

    vehicle_compute_kappa: float = 1.0e-28
    rsu_compute_kappa: float = 1.0e-28
    cloud_compute_kappa: float = 1.0e-28
    vehicle_tx_power_w: float = 0.4
    rsu_receive_power_w: float = 2.0
    cloud_receive_power_w: float = 2.0

    def __post_init__(self) -> None:
        if any(
            not isfinite(value) or value < 0 for value in vars(self).values()
        ):
            raise ValueError("system energy parameters must be finite and non-negative")


@dataclass(frozen=True)
class SlotEnergyLedger:
    """Realized slot energy grouped by the component that owns the cost."""

    vehicle_compute_j: tuple[float, ...]
    vehicle_uplink_j: tuple[float, ...]
    rsu_compute_j: tuple[float, ...]
    rsu_reception_j: tuple[float, ...]
    uav_j: tuple[float, ...]
    cloud_compute_j: float
    cloud_reception_j: float

    @property
    def vehicle_j(self) -> tuple[float, ...]:
        return tuple(
            compute + uplink
            for compute, uplink in zip(
                self.vehicle_compute_j, self.vehicle_uplink_j
            )
        )

    @property
    def rsu_j(self) -> tuple[float, ...]:
        return tuple(
            compute + reception
            for compute, reception in zip(
                self.rsu_compute_j, self.rsu_reception_j
            )
        )

    @property
    def total_j(self) -> float:
        return (
            sum(self.vehicle_j)
            + sum(self.rsu_j)
            + sum(self.uav_j)
            + self.cloud_compute_j
            + self.cloud_reception_j
        )


@dataclass(frozen=True)
class BatteryConfig:
    capacity_j: float
    initial_charge_j: float

    def __post_init__(self) -> None:
        if not isfinite(self.capacity_j) or self.capacity_j <= 0:
            raise ValueError("battery capacity must be finite and positive")
        if not isfinite(self.initial_charge_j):
            raise ValueError("initial charge must be finite")
        if not 0 <= self.initial_charge_j <= self.capacity_j:
            raise ValueError("initial charge must lie within battery capacity")


@dataclass(frozen=True)
class BatteryTransition:
    charge_before_j: float
    expenditure_j: float
    harvest_j: float
    overflow_j: float
    charge_after_j: float


class BatteryState:
    """Physical battery where harvest becomes available after expenditure."""

    def __init__(self, config: BatteryConfig) -> None:
        self.config = config
        self.charge_j = config.initial_charge_j

    def reset(self) -> None:
        self.charge_j = self.config.initial_charge_j

    def can_execute(self, expenditure_j: float, reserve_j: float = 0.0) -> bool:
        self._validate_nonnegative(expenditure_j, "expenditure")
        self._validate_nonnegative(reserve_j, "reserve")
        return expenditure_j <= self.charge_j and (
            self.charge_j - expenditure_j >= reserve_j
        )

    def apply(self, expenditure_j: float, harvest_j: float) -> BatteryTransition:
        self._validate_nonnegative(expenditure_j, "expenditure")
        self._validate_nonnegative(harvest_j, "harvest")
        before = self.charge_j
        if expenditure_j > before + 1e-9:
            raise ValueError("current expenditure cannot use end-of-slot harvest")
        unbounded = before - expenditure_j + harvest_j
        after = max(min(self.config.capacity_j, unbounded), 0.0)
        overflow = max(unbounded - self.config.capacity_j, 0.0)
        self.charge_j = after
        return BatteryTransition(
            charge_before_j=before,
            expenditure_j=expenditure_j,
            harvest_j=harvest_j,
            overflow_j=overflow,
            charge_after_j=after,
        )

    @staticmethod
    def _validate_nonnegative(value: float, name: str) -> None:
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
