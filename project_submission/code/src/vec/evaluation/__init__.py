"""Research-grade scenario, metric, statistics, and evidence pipeline."""

from .artifacts import CORE_METRICS, EvidenceWriter
from .metrics import RunMetrics, SlotTrace, TaskOutcome
from .plots import generate_figures
from .runner import (
    REQUIRED_COMPARISON_METHODS,
    EvaluationMethod,
    EvaluationRun,
    baseline_methods,
    evaluate_scenario,
    missing_required_methods,
)
from .scenarios import (
    SCENARIO_SCHEMA_VERSION,
    STANDARD_STRESSES,
    ScenarioInstance,
    ScenarioManifest,
    ScenarioSpec,
    create_standard_manifest,
    instantiate_scenario,
    load_recorded_mobility_csv,
)
from .statistics import (
    ConfidenceInterval,
    PairedComparison,
    confidence_interval,
    paired_comparison,
)
from .study_plots import generate_study_figures

__all__ = [
    "CORE_METRICS",
    "ConfidenceInterval",
    "EvidenceWriter",
    "EvaluationMethod",
    "EvaluationRun",
    "PairedComparison",
    "REQUIRED_COMPARISON_METHODS",
    "RunMetrics",
    "SCENARIO_SCHEMA_VERSION",
    "STANDARD_STRESSES",
    "ScenarioInstance",
    "ScenarioManifest",
    "ScenarioSpec",
    "SlotTrace",
    "TaskOutcome",
    "baseline_methods",
    "confidence_interval",
    "generate_study_figures",
    "create_standard_manifest",
    "evaluate_scenario",
    "generate_figures",
    "instantiate_scenario",
    "load_recorded_mobility_csv",
    "missing_required_methods",
    "paired_comparison",
]
