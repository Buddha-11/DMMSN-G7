"""Seed-level uncertainty and paired comparisons for research reporting."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, fmean, stdev
from typing import Hashable, Mapping, Sequence


_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class ConfidenceInterval:
    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedComparison:
    method: str
    reference_method: str
    metric: str
    paired_count: int
    mean_method: float
    mean_reference: float
    mean_paired_difference: float
    difference_ci_lower: float
    difference_ci_upper: float
    standardized_paired_effect: float | None
    wins: int
    ties: int
    losses: int
    higher_is_better: bool


def confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> ConfidenceInterval:
    clean = tuple(float(value) for value in values)
    if not clean:
        raise ValueError("confidence interval requires observations")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    mean = fmean(clean)
    if len(clean) == 1:
        return ConfidenceInterval(1, mean, 0.0, 0.0, confidence, mean, mean)
    deviation = stdev(clean)
    error = deviation / sqrt(len(clean))
    critical = _critical_value(len(clean) - 1, confidence)
    half_width = critical * error
    return ConfidenceInterval(
        len(clean),
        mean,
        deviation,
        error,
        confidence,
        mean - half_width,
        mean + half_width,
    )


def paired_comparison(
    method_values: Mapping[Hashable, float],
    reference_values: Mapping[Hashable, float],
    *,
    method: str,
    reference_method: str,
    metric: str,
    higher_is_better: bool,
    confidence: float = 0.95,
) -> PairedComparison:
    method_keys = set(method_values)
    reference_keys = set(reference_values)
    if method_keys != reference_keys:
        missing_method = sorted(reference_keys - method_keys, key=str)
        missing_reference = sorted(method_keys - reference_keys, key=str)
        raise ValueError(
            "paired comparison requires identical scenario keys; "
            f"missing method={missing_method}, missing reference={missing_reference}"
        )
    if not method_keys:
        raise ValueError("paired comparison requires at least one pair")
    ordered = sorted(method_keys, key=str)
    method_samples = [float(method_values[key]) for key in ordered]
    reference_samples = [float(reference_values[key]) for key in ordered]
    differences = [
        method_value - reference_value
        for method_value, reference_value in zip(
            method_samples, reference_samples, strict=True
        )
    ]
    interval = confidence_interval(differences, confidence)
    difference_sd = stdev(differences) if len(differences) > 1 else 0.0
    effect = interval.mean / difference_sd if difference_sd > 0.0 else None
    oriented = differences if higher_is_better else [-value for value in differences]
    tolerance = 1.0e-12
    return PairedComparison(
        method=method,
        reference_method=reference_method,
        metric=metric,
        paired_count=len(ordered),
        mean_method=fmean(method_samples),
        mean_reference=fmean(reference_samples),
        mean_paired_difference=interval.mean,
        difference_ci_lower=interval.lower,
        difference_ci_upper=interval.upper,
        standardized_paired_effect=effect,
        wins=sum(value > tolerance for value in oriented),
        ties=sum(abs(value) <= tolerance for value in oriented),
        losses=sum(value < -tolerance for value in oriented),
        higher_is_better=higher_is_better,
    )


def _critical_value(degrees_of_freedom: int, confidence: float) -> float:
    if confidence == 0.95 and degrees_of_freedom <= 30:
        return _T_975[degrees_of_freedom]
    # Normal approximation is explicit for other confidence levels and for
    # larger samples. Reports retain sample count and standard error.
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)
