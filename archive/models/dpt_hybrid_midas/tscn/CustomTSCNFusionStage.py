import torch
import torch.nn as nn
from transformers.models.dpt.modeling_dpt import DPTFeatureFusionLayer, DPTFeatureFusionStage, DPTForDepthEstimation

class ChannelAttentionModule(nn.Module):  # acts like the SENet
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

class AdaptiveConcatenationModule(nn.Module):
    def __init__(self, in_channels, num_stages):
        super().__init__()
        self.num_stages = num_stages
        self.weights = nn.Parameter(torch.ones(num_stages) / num_stages)
        total_in_channels = in_channels * (num_stages + 1)
        self.channel_attention = ChannelAttentionModule(total_in_channels)
        self.conv = nn.Conv2d(in_channels * (num_stages + 1), in_channels, kernel_size=1)

    def forward(self, decoder_feature, encoder_features):
        # upsample to match decoder feature resolution
        upsampled_features = []
        for feature in encoder_features:
            feature = nn.functional.interpolate(
                feature, size=decoder_feature.shape[2:], mode="bilinear", align_corners=False
            )
            upsampled_features.append(feature)
        
        weights = torch.softmax(self.weights, dim=0)
        weighted_features = [w * f for w, f in zip(weights, upsampled_features)]
        concat_features = torch.cat(weighted_features + [decoder_feature], dim=1)
        
        fused_features = self.channel_attention(concat_features)
        return nn.ReLU()(self.conv(fused_features))

class TransformerSkipConnectionModule(nn.Module):
    def __init__(self, in_channels, num_stages, num_heads=4, num_layers=2, downsample_factor=1):
        super().__init__()
        self.num_stages = num_stages
        self.downsample_factor = downsample_factor
        
        self.total_features = num_stages + 1
        
        # projection to transformer dimension
        self.to_tokens = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Transformer encoder for self-attention across feature maps
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=num_heads,
                dim_feedforward=in_channels * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # projection back to feature map
        self.to_feature = nn.Conv2d(in_channels * self.total_features, in_channels, kernel_size=1)

    def forward(self, decoder_feature, encoder_features):
        if self.downsample_factor > 1:
            decoder_feature = nn.functional.avg_pool2d(decoder_feature, kernel_size=self.downsample_factor)
            encoder_features = [
                nn.functional.avg_pool2d(feature, kernel_size=self.downsample_factor)
                for feature in encoder_features
            ]
        
        # upsample encoder features to match decoder feature resolution
        upsampled_features = []
        for feature in encoder_features:
            feature = nn.functional.interpolate(
                feature, size=decoder_feature.shape[2:], mode="bilinear", align_corners=False
            )
            upsampled_features.append(feature)
        
        # combine encoder + decoder features
        all_features = upsampled_features + [decoder_feature]
        batch_size, C, H, W = all_features[0].shape
        
        # project and flatten features to tokens
        tokens = torch.stack([self.to_tokens(f).flatten(2).permute(0, 2, 1) for f in all_features], dim=1)
        tokens = tokens.view(batch_size, self.total_features * H * W, C)
        
        transformed = self.transformer(tokens)
        
        transformed = transformed.view(batch_size, self.total_features, H * W, C).permute(0, 3, 1, 2)
        transformed = transformed.reshape(batch_size, C * self.total_features, H, W)
        
        fused_feature = self.to_feature(transformed)
        
        if self.downsample_factor > 1:
            fused_feature = nn.functional.interpolate(
                fused_feature, scale_factor=self.downsample_factor, mode="bilinear", align_corners=False
            )
        
        return fused_feature

class CustomFeatureFusionLayer(DPTFeatureFusionLayer):
    def __init__(self, config, align_corners=True):
        super().__init__(config, align_corners)
        self.tscn = TransformerSkipConnectionModule(
            in_channels=config.fusion_hidden_size, num_stages=4, downsample_factor=2
        )

    def forward(self, hidden_state, encoder_features):
        # use TSCN to fuse all encoder features with the current decoder feature
        hidden_state = self.tscn(hidden_state, encoder_features)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state

class CustomTSCNFusionStage(DPTFeatureFusionStage):
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
