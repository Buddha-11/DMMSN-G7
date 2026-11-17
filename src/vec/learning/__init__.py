"""Optional PyTorch learning stack for the queue-aware VEC simulator.

Import this subpackage explicitly; :mod:`src.vec` remains usable for simulator
and baseline experiments on machines without PyTorch.
"""

from .adapter import EncodedControlObservation, VecPolicyAdapter
from .agent import HybridMADDPG, HybridMADDPGConfig, UpdateDiagnostics
from .controller import HybridPolicyController
from .meta import (
    AdaptationResult,
    BoundedOnlineAdapter,
    OnlineAdaptationReport,
    ReptileMetaLearner,
    vec_physical_state_token,
)
from .mappo import (
    HybridMAPPO,
    HybridMAPPOActor,
    HybridMAPPOConfig,
    MAPPORolloutStep,
    MAPPOUpdateDiagnostics,
)
from .mappo_controller import HybridMAPPOController
from .networks import (
    CentralizedCritic,
    HybridActionRepair,
    HybridRoleActor,
    MaskedTaskSetEncoder,
    RoleSpec,
)
from .replay import HybridTransition, TaskLocalReplayBank

__all__ = [
    "AdaptationResult",
    "BoundedOnlineAdapter",
    "CentralizedCritic",
    "EncodedControlObservation",
    "HybridActionRepair",
    "HybridMADDPG",
    "HybridMADDPGConfig",
    "HybridMAPPO",
    "HybridMAPPOActor",
    "HybridMAPPOConfig",
    "HybridMAPPOController",
    "HybridPolicyController",
    "HybridRoleActor",
    "HybridTransition",
    "MaskedTaskSetEncoder",
    "MAPPORolloutStep",
    "MAPPOUpdateDiagnostics",
    "OnlineAdaptationReport",
    "ReptileMetaLearner",
    "RoleSpec",
    "TaskLocalReplayBank",
    "UpdateDiagnostics",
    "VecPolicyAdapter",
    "vec_physical_state_token",
]
