"""
Models Package for SimAM-KD Framework
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)
"""

from .attention import (
    SimAM,
    CoordinateAttention,
    ParallelSimCA,
    SEBlock,
    CBAM,
    get_attention_module,
)

from .student import (
    MobileNetV3SimAM,
    mobilenetv3_simam_small,
    mobilenetv3_simam_large,
)

from .teacher import (
    TeacherResNet,
    TeacherEfficientNet,
    WideResNet,
    get_teacher_model,
)

__all__ = [
    # Attention modules
    'SimAM',
    'CoordinateAttention',
    'ParallelSimCA',
    'SEBlock',
    'CBAM',
    'get_attention_module',
    # Student models
    'MobileNetV3SimAM',
    'mobilenetv3_simam_small',
    'mobilenetv3_simam_large',
    # Teacher models
    'TeacherResNet',
    'TeacherEfficientNet',
    'WideResNet',
    'get_teacher_model',
]
