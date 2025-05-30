import torch
import torch.nn as nn
from transformers.models.dpt.modeling_dpt import DPTFeatureFusionLayer, DPTPreActResidualLayer

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        weights = self.mlp(avg).view(x.size(0), x.size(1), 1, 1)
        return x * weights

class CustomFeatureFusionLayer(DPTFeatureFusionLayer):
    def __init__(self, config, align_corners=True):
        super().__init__(config, align_corners)
        self.attention = ChannelAttentionModule(in_channels=config.fusion_hidden_size)

    def forward(self, hidden_state, residual=None):
        hidden_state = self.attention(hidden_state)
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), 
                    mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)
        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state
