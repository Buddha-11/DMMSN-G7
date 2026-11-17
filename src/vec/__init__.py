"""Queue-aware vehicular edge-computing simulator.

This package is intentionally separate from :mod:`src.env`, which implements
the repository's earlier diffusion-MAPPO experiment.  The modules here follow
the task-persistent queueing model described in ``README_PROJECT.md``.
"""

from .env import (
    Destination,
    DispatchDecision,
    ScheduledArrival,
    ScheduledHarvest,
    StepAction,
    StepResult,
    UAVMode,
    VecEnvConfig,
    VecQueueEnv,
)
from .energy import (
    BatteryConfig,
    BatteryState,
    BatteryTransition,
    RotaryWingEnergyConfig,
    SlotEnergyDemand,
    SlotEnergyLedger,
    SystemEnergyConfig,
    UAVEnergyModel,
)
from .control import (
    AgentObservation,
    ControlObservation,
    DestinationMask,
    DispatchTrace,
    ExecutedActionTrace,
)
from .baselines import (
    BaselineDecision,
    BaselineDiagnostics,
    BaselineRollout,
    BaselineRolloutMetrics,
    BaselineStep,
    DriftCandidateEvaluator,
    GeneticDriftController,
    GreedyConfig,
    LocalOnlyController,
    NearestFeasibleEdgeController,
    QueueEnergyGreedyController,
    RandomFeasibleController,
    execute_controller_step,
    run_controller_episode,
)
from .lyapunov import (
    LyapunovConfig,
    LyapunovScore,
    LyapunovSlotInput,
    LyapunovTracker,
)
from .mobility import (
    PolylineRoute,
    RecordedTraceMobility,
    SyntheticRoadMobility,
    VehicleRouteConfig,
)
from .radio import DeterministicChannelModel, QueueAwareOFDMA, RadioConfig
from .return_plan import ReturnLandingPlanner, ReturnPlan
from .safety import (
    FlightLimits,
    FlightSafetyController,
    MotionCommand,
    MotionRepairResult,
    UAVKinematicState,
)
from .tasks import DestinationKind, TaskRecord, TaskSpec, TaskState

__all__ = [
    "Destination",
    "AgentObservation",
    "ControlObservation",
    "DestinationMask",
    "DispatchTrace",
    "ExecutedActionTrace",
    "DestinationKind",
    "DeterministicChannelModel",
    "BatteryConfig",
    "BatteryState",
    "BatteryTransition",
    "BaselineDecision",
    "BaselineDiagnostics",
    "BaselineRollout",
    "BaselineRolloutMetrics",
    "BaselineStep",
    "DispatchDecision",
    "DriftCandidateEvaluator",
    "ScheduledArrival",
    "ScheduledHarvest",
    "PolylineRoute",
    "FlightLimits",
    "FlightSafetyController",
    "GeneticDriftController",
    "GreedyConfig",
    "LocalOnlyController",
    "MotionCommand",
    "MotionRepairResult",
    "NearestFeasibleEdgeController",
    "LyapunovConfig",
    "LyapunovScore",
    "LyapunovSlotInput",
    "LyapunovTracker",
    "QueueAwareOFDMA",
    "QueueEnergyGreedyController",
    "RandomFeasibleController",
    "RadioConfig",
    "RecordedTraceMobility",
    "ReturnLandingPlanner",
    "ReturnPlan",
    "RotaryWingEnergyConfig",
    "SlotEnergyDemand",
    "SlotEnergyLedger",
    "SystemEnergyConfig",
    "StepAction",
    "StepResult",
    "UAVMode",
    "TaskRecord",
    "TaskSpec",
    "TaskState",
    "SyntheticRoadMobility",
    "VecEnvConfig",
    "VecQueueEnv",
    "VehicleRouteConfig",
    "UAVEnergyModel",
    "UAVKinematicState",
    "execute_controller_step",
    "run_controller_episode",
]
