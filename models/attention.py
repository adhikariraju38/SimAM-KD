"""
Attention Modules for SimAM-KD Framework
Authors: Raju Kumar Yadav, Rajesh Khanal, Safalta Kumari Yadav, Rikesh Kumar Shah, Bibek Kumar Gupta

Paper: SimAM-KD: Attention-Enhanced Knowledge Distillation for Efficient Image Classification
GitHub: https://github.com/adhikariraju38/SimAM-KD

This module implements various attention mechanisms including:
- SimAM: Simple, Parameter-Free Attention Module
- CA: Coordinate Attention
- C//Sim: Parallel combination of CA and SimAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    """
    Simple, Parameter-Free Attention Module (SimAM)

    Reference: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
               for Convolutional Neural Networks" (ICML 2021)

    SimAM computes 3D attention weights without adding any parameters.
    It uses energy functions based on neuroscience theories.
    """

    def __init__(self, e_lambda: float = 1e-4):
        """
        Args:
            e_lambda: Regularization parameter to avoid division by zero
        """
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SimAM attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-weighted tensor of same shape
        """
        b, c, h, w = x.size()
        n = w * h - 1  # Number of neurons minus 1

        # Compute variance across spatial dimensions
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # Compute attention weights using energy function
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * torch.sigmoid(y)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CA) Module

    Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design"
               (CVPR 2021)

    CA captures long-range dependencies along one spatial direction and preserves
    precise positional information along the other spatial direction.
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for intermediate layers
        """
        super(CoordinateAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)  # Swish activation

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Coordinate Attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-weighted tensor of same shape
        """
        identity = x
        b, c, h, w = x.size()

        # Encode channel information along height and width directions
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1) -> (B, C, 1, W) permuted

        # Concatenate and apply shared transformation
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split and generate attention maps
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class ParallelSimCA(nn.Module):
    """
    Parallel combination of SimAM and Coordinate Attention (C//Sim)

    This module applies both SimAM and CA in parallel and combines their outputs.
    Inspired by MobileNetV3-C//Sim architecture.
    """

    def __init__(self, in_channels: int, reduction: int = 32, e_lambda: float = 1e-4):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio for CA
            e_lambda: Regularization parameter for SimAM
        """
        super(ParallelSimCA, self).__init__()

        self.simam = SimAM(e_lambda=e_lambda)
        self.ca = CoordinateAttention(in_channels, reduction)

        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining SimAM and CA outputs.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Combined attention-weighted tensor
        """
        simam_out = self.simam(x)
        ca_out = self.ca(x)

        # Weighted combination with learnable alpha
        alpha = torch.sigmoid(self.alpha)
        return alpha * simam_out + (1 - alpha) * ca_out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Baseline comparison)

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Baseline comparison)

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        # Spatial attention
        self.conv_spatial = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))

        return x * spatial_att


def get_attention_module(attention_type: str, in_channels: int, **kwargs) -> nn.Module:
    """
    Factory function to create attention modules.

    Args:
        attention_type: Type of attention ('simam', 'ca', 'parallel', 'se', 'cbam', 'none')
        in_channels: Number of input channels
        **kwargs: Additional arguments for specific attention types

    Returns:
        Attention module or Identity if 'none'
    """
    attention_types = {
        'simam': lambda: SimAM(e_lambda=kwargs.get('e_lambda', 1e-4)),
        'ca': lambda: CoordinateAttention(in_channels, reduction=kwargs.get('reduction', 32)),
        'parallel': lambda: ParallelSimCA(
            in_channels,
            reduction=kwargs.get('reduction', 32),
            e_lambda=kwargs.get('e_lambda', 1e-4)
        ),
        'se': lambda: SEBlock(in_channels, reduction=kwargs.get('reduction', 16)),
        'cbam': lambda: CBAM(in_channels, reduction=kwargs.get('reduction', 16)),
        'none': lambda: nn.Identity()
    }

    if attention_type not in attention_types:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Choose from {list(attention_types.keys())}")

    return attention_types[attention_type]()


if __name__ == "__main__":
    # Test attention modules
    print("Testing attention modules...")

    x = torch.randn(2, 64, 32, 32)

    for att_type in ['simam', 'ca', 'parallel', 'se', 'cbam', 'none']:
        att = get_attention_module(att_type, 64)
        out = att(x)
        print(f"{att_type.upper():10s} - Input: {x.shape} -> Output: {out.shape}")

        # Count parameters
        params = sum(p.numel() for p in att.parameters())
        print(f"           Parameters: {params}")
