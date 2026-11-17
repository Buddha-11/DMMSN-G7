"""Reproducible training orchestration for the VEC learning stack."""

from .config import TRAINING_SCHEMA_VERSION, TrainingConfig
from .mappo_trainer import (
    MAPPO_TRAINING_SCHEMA_VERSION,
    MAPPOTrainingConfig,
    VecMAPPOTrainer,
)
from .trainer import TrainingResult, VecTrainer, explore_hybrid_actions

__all__ = [
    "TRAINING_SCHEMA_VERSION",
    "TrainingConfig",
    "TrainingResult",
    "MAPPO_TRAINING_SCHEMA_VERSION",
    "MAPPOTrainingConfig",
    "VecMAPPOTrainer",
    "VecTrainer",
    "explore_hybrid_actions",
]
