"""
Teacher Models for Knowledge Distillation
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

This module provides pre-trained teacher models (ResNet, EfficientNet) adapted
for CIFAR-10/100 datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class TeacherResNet(nn.Module):
    """
    ResNet teacher model adapted for CIFAR-sized inputs.

    Modifications for CIFAR (32x32):
    - First conv kernel: 7x7 -> 3x3
    - First conv stride: 2 -> 1
    - Remove max pooling layer
    """

    def __init__(
        self,
        num_classes: int = 10,
        variant: str = 'resnet50',  # resnet18, resnet34, resnet50, resnet101
        pretrained: bool = True,
        input_size: int = 32,
    ):
        super(TeacherResNet, self).__init__()

        # Load pre-trained model
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }

        if variant not in model_dict:
            raise ValueError(f"Unknown variant: {variant}")

        weights = 'IMAGENET1K_V1' if pretrained else None
        base_model = model_dict[variant](weights=weights)

        # Modify for CIFAR-sized inputs
        if input_size <= 64:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Copy weights from original conv (center crop)
            if pretrained:
                with torch.no_grad():
                    self.conv1.weight.copy_(base_model.conv1.weight[:, :, 2:5, 2:5])
        else:
            self.conv1 = base_model.conv1

        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        # Skip max pooling for small inputs
        self.use_maxpool = input_size > 64
        if self.use_maxpool:
            self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # New classifier
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.use_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class TeacherEfficientNet(nn.Module):
    """
    EfficientNet teacher model adapted for CIFAR-sized inputs.
    """

    def __init__(
        self,
        num_classes: int = 10,
        variant: str = 'efficientnet_b0',
        pretrained: bool = True,
        input_size: int = 32,
    ):
        super(TeacherEfficientNet, self).__init__()

        # Load pre-trained model
        model_dict = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
        }

        if variant not in model_dict:
            raise ValueError(f"Unknown variant: {variant}")

        weights = 'IMAGENET1K_V1' if pretrained else None
        base_model = model_dict[variant](weights=weights)

        # Features
        self.features = base_model.features

        # Modify first conv for small inputs
        if input_size <= 64:
            old_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                3, old_conv.out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # New classifier
        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class WideResNet(nn.Module):
    """
    Wide ResNet for CIFAR - commonly used as a strong teacher.

    WRN-40-2 and WRN-28-10 are popular choices.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.3,
    ):
        super(WideResNet, self).__init__()

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(n_channels[0], n_channels[1], n, 1, dropout_rate)
        self.block2 = self._make_layer(n_channels[1], n_channels[2], n, 2, dropout_rate)
        self.block3 = self._make_layer(n_channels[2], n_channels[3], n, 2, dropout_rate)

        self.bn = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channels[3], num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, n_blocks, stride, dropout_rate):
        layers = []
        layers.append(WideBasicBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, n_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class WideBasicBlock(nn.Module):
    """Basic block for Wide ResNet."""

    def __init__(self, in_channels, out_channels, stride, dropout_rate):
        super(WideBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                     stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + shortcut


def get_teacher_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    input_size: int = 32,
) -> nn.Module:
    """
    Factory function to create teacher models.

    Args:
        model_name: Name of the teacher model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        input_size: Input image size

    Returns:
        Teacher model
    """
    if model_name.startswith('resnet'):
        return TeacherResNet(num_classes, model_name, pretrained, input_size)
    elif model_name.startswith('efficientnet'):
        return TeacherEfficientNet(num_classes, model_name, pretrained, input_size)
    elif model_name.startswith('wrn'):
        # Parse WRN-depth-width format
        parts = model_name.split('-')
        depth = int(parts[1]) if len(parts) > 1 else 28
        width = int(parts[2]) if len(parts) > 2 else 10
        return WideResNet(num_classes, depth, width)
    else:
        raise ValueError(f"Unknown teacher model: {model_name}")


if __name__ == "__main__":
    # Test teacher models
    print("Testing teacher models...")

    x = torch.randn(2, 3, 32, 32)

    models_to_test = ['resnet18', 'resnet50', 'efficientnet_b0', 'wrn-28-10']

    for model_name in models_to_test:
        model = get_teacher_model(model_name, num_classes=10, pretrained=False)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{model_name:20s} | Params: {params:.2f}M | Output: {out.shape}")
