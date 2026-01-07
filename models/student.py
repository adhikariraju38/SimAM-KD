"""
Student Model: MobileNetV3 with SimAM Integration
Authors: Raju Kumar Yadav, Rajesh Khanal, Safalta Kumari Yadav, Rikesh Kumar Shah, Bibek Kumar Gupta

Paper: SimAM-KD: Attention-Enhanced Knowledge Distillation for Efficient Image Classification
GitHub: https://github.com/adhikariraju38/SimAM-KD

This module implements MobileNetV3 variants with optional attention mechanisms
for the SimAM-KD framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable
from functools import partial

from .attention import get_attention_module


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    Ensure the number of channels is divisible by divisor.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function."""

    def __init__(self, inplace: bool = True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3, inplace=self.inplace) / 6


class HardSwish(nn.Module):
    """Hard Swish activation function."""

    def __init__(self, inplace: bool = True):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class ConvBNActivation(nn.Sequential):
    """Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias=False
            ),
            norm_layer(out_channels),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation module for MobileNetV3."""

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return x * scale


class InvertedResidualConfig:
    """Configuration for Inverted Residual block."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
    ):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block with optional attention."""

    def __init__(
        self,
        config: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        attention_type: str = 'none',
        attention_kwargs: dict = None,
    ):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (
            config.stride == 1 and config.in_channels == config.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = HardSwish if config.use_hs else nn.ReLU

        # Expansion
        if config.expanded_channels != config.in_channels:
            layers.append(
                ConvBNActivation(
                    config.in_channels,
                    config.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise convolution
        layers.append(
            ConvBNActivation(
                config.expanded_channels,
                config.expanded_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                groups=config.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # Squeeze-and-Excitation (original MobileNetV3 SE)
        if config.use_se:
            layers.append(SqueezeExcitation(config.expanded_channels))

        # Add custom attention after SE (SimAM, CA, etc.)
        if attention_type != 'none':
            attention_kwargs = attention_kwargs or {}
            layers.append(
                get_attention_module(attention_type, config.expanded_channels, **attention_kwargs)
            )

        # Projection
        layers.append(
            ConvBNActivation(
                config.expanded_channels,
                config.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = config.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


class MobileNetV3SimAM(nn.Module):
    """
    MobileNetV3 with SimAM Attention Integration

    This model adds attention mechanisms (SimAM, CA, or Parallel) to MobileNetV3
    for improved feature extraction.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        variant: str = 'small',  # 'small' or 'large'
        attention_type: str = 'simam',  # 'simam', 'ca', 'parallel', 'se', 'cbam', 'none'
        attention_kwargs: dict = None,
        dropout: float = 0.2,
        input_size: int = 32,  # CIFAR: 32, ImageNet: 224
    ):
        """
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier for channels
            variant: 'small' or 'large' variant
            attention_type: Type of attention to add
            attention_kwargs: Additional args for attention module
            dropout: Dropout rate before classifier
            input_size: Input image size (affects first conv stride)
        """
        super(MobileNetV3SimAM, self).__init__()

        self.attention_type = attention_type
        self.attention_kwargs = attention_kwargs or {}

        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # Get configuration based on variant
        if variant == 'small':
            inverted_residual_setting = self._get_small_config(width_mult)
            last_channel = _make_divisible(576 * width_mult, 8)
        else:
            inverted_residual_setting = self._get_large_config(width_mult)
            last_channel = _make_divisible(960 * width_mult, 8)

        # First convolution
        first_conv_out = _make_divisible(16 * width_mult, 8)
        first_stride = 1 if input_size <= 64 else 2  # Adjust for CIFAR

        self.features = nn.Sequential()
        self.features.add_module(
            'conv_first',
            ConvBNActivation(
                3, first_conv_out, kernel_size=3, stride=first_stride,
                norm_layer=norm_layer, activation_layer=HardSwish
            )
        )

        # Build inverted residual blocks
        for idx, config in enumerate(inverted_residual_setting):
            # Adjust stride for small input sizes
            if input_size <= 64 and config.stride == 2:
                # Reduce strides for CIFAR-sized inputs
                if idx in [0, 1]:  # Keep some downsampling
                    pass
                elif idx > 6:
                    config.stride = 1

            block = InvertedResidual(
                config, norm_layer, attention_type, self.attention_kwargs
            )
            self.features.add_module(f'block_{idx}', block)

        # Last convolution
        last_conv_in = inverted_residual_setting[-1].out_channels
        self.features.add_module(
            'conv_last',
            ConvBNActivation(
                last_conv_in, last_channel, kernel_size=1,
                norm_layer=norm_layer, activation_layer=HardSwish
            )
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 1280),
            HardSwish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _get_small_config(self, width_mult: float) -> List[InvertedResidualConfig]:
        """MobileNetV3-Small configuration."""
        bneck_conf = partial(InvertedResidualConfig)
        adjust_channels = partial(_make_divisible, divisor=8)

        return [
            bneck_conf(adjust_channels(16 * width_mult), 3, adjust_channels(16 * width_mult),
                      adjust_channels(16 * width_mult), True, "RE", 2),
            bneck_conf(adjust_channels(16 * width_mult), 3, adjust_channels(72 * width_mult),
                      adjust_channels(24 * width_mult), False, "RE", 2),
            bneck_conf(adjust_channels(24 * width_mult), 3, adjust_channels(88 * width_mult),
                      adjust_channels(24 * width_mult), False, "RE", 1),
            bneck_conf(adjust_channels(24 * width_mult), 5, adjust_channels(96 * width_mult),
                      adjust_channels(40 * width_mult), True, "HS", 2),
            bneck_conf(adjust_channels(40 * width_mult), 5, adjust_channels(240 * width_mult),
                      adjust_channels(40 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(40 * width_mult), 5, adjust_channels(240 * width_mult),
                      adjust_channels(40 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(40 * width_mult), 5, adjust_channels(120 * width_mult),
                      adjust_channels(48 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(48 * width_mult), 5, adjust_channels(144 * width_mult),
                      adjust_channels(48 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(48 * width_mult), 5, adjust_channels(288 * width_mult),
                      adjust_channels(96 * width_mult), True, "HS", 2),
            bneck_conf(adjust_channels(96 * width_mult), 5, adjust_channels(576 * width_mult),
                      adjust_channels(96 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(96 * width_mult), 5, adjust_channels(576 * width_mult),
                      adjust_channels(96 * width_mult), True, "HS", 1),
        ]

    def _get_large_config(self, width_mult: float) -> List[InvertedResidualConfig]:
        """MobileNetV3-Large configuration."""
        bneck_conf = partial(InvertedResidualConfig)
        adjust_channels = partial(_make_divisible, divisor=8)

        return [
            bneck_conf(adjust_channels(16 * width_mult), 3, adjust_channels(16 * width_mult),
                      adjust_channels(16 * width_mult), False, "RE", 1),
            bneck_conf(adjust_channels(16 * width_mult), 3, adjust_channels(64 * width_mult),
                      adjust_channels(24 * width_mult), False, "RE", 2),
            bneck_conf(adjust_channels(24 * width_mult), 3, adjust_channels(72 * width_mult),
                      adjust_channels(24 * width_mult), False, "RE", 1),
            bneck_conf(adjust_channels(24 * width_mult), 5, adjust_channels(72 * width_mult),
                      adjust_channels(40 * width_mult), True, "RE", 2),
            bneck_conf(adjust_channels(40 * width_mult), 5, adjust_channels(120 * width_mult),
                      adjust_channels(40 * width_mult), True, "RE", 1),
            bneck_conf(adjust_channels(40 * width_mult), 5, adjust_channels(120 * width_mult),
                      adjust_channels(40 * width_mult), True, "RE", 1),
            bneck_conf(adjust_channels(40 * width_mult), 3, adjust_channels(240 * width_mult),
                      adjust_channels(80 * width_mult), False, "HS", 2),
            bneck_conf(adjust_channels(80 * width_mult), 3, adjust_channels(200 * width_mult),
                      adjust_channels(80 * width_mult), False, "HS", 1),
            bneck_conf(adjust_channels(80 * width_mult), 3, adjust_channels(184 * width_mult),
                      adjust_channels(80 * width_mult), False, "HS", 1),
            bneck_conf(adjust_channels(80 * width_mult), 3, adjust_channels(184 * width_mult),
                      adjust_channels(80 * width_mult), False, "HS", 1),
            bneck_conf(adjust_channels(80 * width_mult), 3, adjust_channels(480 * width_mult),
                      adjust_channels(112 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(112 * width_mult), 3, adjust_channels(672 * width_mult),
                      adjust_channels(112 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(112 * width_mult), 5, adjust_channels(672 * width_mult),
                      adjust_channels(160 * width_mult), True, "HS", 2),
            bneck_conf(adjust_channels(160 * width_mult), 5, adjust_channels(960 * width_mult),
                      adjust_channels(160 * width_mult), True, "HS", 1),
            bneck_conf(adjust_channels(160 * width_mult), 5, adjust_channels(960 * width_mult),
                      adjust_channels(160 * width_mult), True, "HS", 1),
        ]

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (for knowledge distillation)."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def mobilenetv3_simam_small(num_classes: int = 10, attention_type: str = 'simam', **kwargs):
    """MobileNetV3-Small with SimAM attention."""
    return MobileNetV3SimAM(
        num_classes=num_classes,
        variant='small',
        attention_type=attention_type,
        **kwargs
    )


def mobilenetv3_simam_large(num_classes: int = 10, attention_type: str = 'simam', **kwargs):
    """MobileNetV3-Large with SimAM attention."""
    return MobileNetV3SimAM(
        num_classes=num_classes,
        variant='large',
        attention_type=attention_type,
        **kwargs
    )


if __name__ == "__main__":
    # Test models
    print("Testing MobileNetV3 with SimAM...")

    for variant in ['small', 'large']:
        for att in ['none', 'simam', 'ca', 'parallel']:
            model = MobileNetV3SimAM(
                num_classes=10,
                variant=variant,
                attention_type=att,
                input_size=32
            )

            x = torch.randn(2, 3, 32, 32)
            out = model(x)

            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"MobileNetV3-{variant.capitalize()} + {att.upper():8s} | "
                  f"Params: {params:.2f}M | Output: {out.shape}")
