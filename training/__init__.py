"""
Training Package for SimAM-KD Framework
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)
"""

from .distillation import (
    DistillationLoss,
    FeatureDistillationLoss,
    DistillationTrainer,
    train_teacher,
)

from .pruning import (
    StructuredPruner,
    ManualChannelPruner,
    count_parameters,
    count_flops,
    get_model_size_mb,
    prune_model_simple,
)

__all__ = [
    'DistillationLoss',
    'FeatureDistillationLoss',
    'DistillationTrainer',
    'train_teacher',
    'StructuredPruner',
    'ManualChannelPruner',
    'count_parameters',
    'count_flops',
    'get_model_size_mb',
    'prune_model_simple',
]
