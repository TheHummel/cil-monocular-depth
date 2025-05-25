import torch
import torch.nn as nn
from transformers.models.dpt.modeling_dpt import DPTFeatureFusionLayer, DPTFeatureFusionStage, DPTForDepthEstimation


class ChannelAttentionModule(nn.Module):  # acts like the SENet
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        weights = self.mlp(avg) + self.mlp(max_out)
        weights = weights.view(x.size(0), x.size(1), 1, 1)
        return x * weights

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class AdaptiveConcatenationModule(nn.Module):
    def __init__(self, in_channels, num_stages):
        super().__init__()
        self.num_stages = num_stages
        total_in_channels = in_channels * (num_stages + 1)

        self.adaptive_weights = nn.Sequential(
            nn.Conv2d(total_in_channels, num_stages, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = ChannelAttentionModule(total_in_channels)
        self.spatial_attention = SpatialAttentionModule()

        self.intermediate_conv = nn.Sequential(
            nn.Conv2d(total_in_channels, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        
        self.conv = nn.Conv2d(512, in_channels, kernel_size=1)

    def forward(self, decoder_feature, encoder_features):
        # upsample to match decoder feature resolution
        upsampled_features = []
        for feature in encoder_features:
            feature = nn.functional.interpolate(
                feature, size=decoder_feature.shape[2:], mode="bicubic", align_corners=False
            )
            spatial_weight = self.spatial_attention(feature)
            feature = feature * spatial_weight
            upsampled_features.append(feature)
        
        concat_features = torch.cat(upsampled_features + [decoder_feature], dim=1)
        weights = self.adaptive_weights(concat_features)
        weights = torch.softmax(weights, dim=1)
        weights = weights.split(1, dim=1)
        weighted_features = [w * f for w, f in zip(weights, upsampled_features)]
        concat_features = torch.cat(weighted_features + [decoder_feature], dim=1)
        
        # CBAM-like attention
        channel_features = self.channel_attention(concat_features)
        channel_features = concat_features + channel_features
        spatial_weights = self.spatial_attention(channel_features)
        fused_features = channel_features * spatial_weights
        fused_features = channel_features + fused_features
        
        fused_features = self.intermediate_conv(fused_features)
        return nn.ReLU()(self.conv(fused_features))

class CustomFeatureFusionLayer(DPTFeatureFusionLayer):
    def __init__(self, config, align_corners=True):
        super().__init__(config, align_corners)
        self.acm = AdaptiveConcatenationModule(
            in_channels=config.fusion_hidden_size, num_stages=4  # 4 encoder stages: layers 3, 6, 9, 12
        )

    def forward(self, hidden_state, encoder_features):
        # use ACM to fuse all encoder features with the current decoder feature
        hidden_state = self.acm(hidden_state, encoder_features)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state

class CustomFSCNFusionStage(DPTFeatureFusionStage):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([CustomFeatureFusionLayer(config) for _ in range(4)])

    def forward(self, features):
        hidden_states = features[::-1]

        fused_hidden_states = []
        fused_hidden_state = None
        for hidden_state, layer in zip(hidden_states, self.layers):
            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = hidden_state
            # pass all encoder features to each layer
            fused_hidden_state = layer(fused_hidden_state, features)
            fused_hidden_states.append(fused_hidden_state)
        return fused_hidden_states