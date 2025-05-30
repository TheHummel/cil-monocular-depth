#!/usr/bin/env python
# coding: utf-8

# # Prep
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DPTForDepthEstimation

from data.dataset import DepthDataset
from utils.helpers import (
    ensure_dir,
    custom_collate_fn,
)
from training.train import train_model
from training.loss import SILogLoss
from inference.evaluate import evaluate_model, generate_test_predictions

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/cluster/courses/cil/monocular_depth/data"
train_dir = os.path.join(data_dir, "train/")
test_dir = os.path.join(data_dir, "test/")
train_list_file = os.path.join(data_dir, "train_list.txt")
test_list_file = os.path.join(data_dir, "test_list.txt")
output_dir = os.path.join(current_dir, "data/monocular_depth_output")
results_dir = os.path.join(output_dir, "results")
predictions_dir = os.path.join(output_dir, "predictions")

print(f"current_dir: {current_dir}")
print(f"output_dir: {output_dir}")

# ### Hyperparameters

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (426, 560)
NUM_WORKERS = 2
PIN_MEMORY = True

# # Model 

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
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
        total_in_channels = 128 * num_stages + in_channels
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(total_in_channels , num_stages, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.channel_attention = ChannelAttentionModule(total_in_channels)
        self.spatial_attention = SpatialAttentionModule()
        self.intermediate_conv = nn.Sequential(
            nn.Conv2d(total_in_channels , 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.edge_preservation = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, bias=False
        )
        with torch.no_grad():
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            )
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            )
            for i in range(512):
                self.edge_preservation.weight[i, i, :, :] = (
                    sobel_x if i % 2 == 0 else sobel_y
                )
        self.conv = nn.Conv2d(512, in_channels, kernel_size=1)

    def forward(self, decoder_feature, encoder_features):
        upsampled_features = []
        for feature in encoder_features:
            feature = nn.functional.interpolate(
                feature,
                size=decoder_feature.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            spatial_weight = self.spatial_attention(feature)
            feature = feature * spatial_weight
            upsampled_features.append(feature)

        concat_features = torch.cat(upsampled_features + [decoder_feature], dim=1)

        weights = self.weight_predictor(concat_features)
        weights = torch.softmax(weights, dim=1)

        weights = weights.split(1, dim=1)
        weighted_features = [w * f for w, f in zip(weights, upsampled_features)]
        concat_features = torch.cat(weighted_features + [decoder_feature], dim=1)
  
        channel_features = self.channel_attention(concat_features)
        channel_features = concat_features + channel_features

        spatial_weights = self.spatial_attention(channel_features)
        fused_features = channel_features * spatial_weights
        fused_features = channel_features + fused_features
        fused_features = self.intermediate_conv(fused_features)
        edge_features = self.edge_preservation(fused_features)
        fused_features = fused_features + edge_features
        return nn.ReLU()(self.conv(fused_features))


class EnhancedUNet(nn.Module):
    def __init__(self):
        super(EnhancedUNet, self).__init__()

        # Get pretrained encoder part of DPT-MiDaS
        self.midas = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        for param in self.midas.parameters():
            param.requires_grad = False

        # Encoder blocks
        self.enc1 = UNetBlock(3, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.enc4 = UNetBlock(128, 256)
      
        # Decoder blocks
        self.dec4 = UNetBlock(256 + 128, 128)
        self.dec3 = UNetBlock(128 + 64, 64)
        self.dec2 = UNetBlock(64 + 32, 32)
        self.dec1 = UNetBlock(32, 32)

        # Fuse 4 DPT MiDaS features (hidden_states 3,6,9,12)
        self.f1 = AdaptiveConcatenationModule(
            in_channels=256, num_stages=4
        )  # 256 channels (enc4)
        self.f2 = AdaptiveConcatenationModule(
            in_channels=128, num_stages=4
        )  # 128 channels (after dec4)
        self.f3 = AdaptiveConcatenationModule(
            in_channels=64, num_stages=4
        )  # 64 channels (after dec3)
        self.f4 = AdaptiveConcatenationModule(
            in_channels=32, num_stages=4
        )  # 32 channels (after dec2)

        # Final layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        # projection applied to the MiDaS extraced features to reduce parameter count
        self.midas_feature_proj4 = nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.midas_feature_proj3 = nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.midas_feature_proj2 = nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1)
        self.midas_feature_proj1 = nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Compute features from dpt-hybrid-midas
        with torch.no_grad():
            # take midas encoder features at different stages (from layers 3, 6, 9, 12)
            # Note: hidden_states[0] is the input embedding
            hidden_states = self.midas.dpt(x, output_hidden_states=True).hidden_states
            hidden_states = [hidden_states[3], hidden_states[6], hidden_states[9], hidden_states[12]] # Tensors of shape (batch_size, 577, 768)

            midas_features = []
            for hidden_state in hidden_states:
                feature = hidden_state[:, 1:, :]
                cls = hidden_state[:, 0, :]
                cls = cls.view(-1, 1, cls.shape[1])
                cls = cls.expand(-1, 576, -1)
                feature = feature + cls
                feature = feature.permute(0, 2, 1).reshape(x.shape[0], 768, 24, 24) # now has shape (batch_size, 768, 24, 24)
                midas_features.append(feature) 

        # Ppoject the 4 midas encoder features to lower channel dimension
        mf1 = self.midas_feature_proj1(midas_features[0])
        mf2 = self.midas_feature_proj2(midas_features[1])
        mf3 = self.midas_feature_proj3(midas_features[2])
        mf4 = self.midas_feature_proj4(midas_features[3])

        dpt_features = [mf1, mf2, mf3, mf4] 

        # U-Net Encoder
        enc1 = self.enc1(x)  # [batch_size, 32, 192, 192]
        x = self.pool(enc1)  # [batch_size, 32, 96, 96]

        enc2 = self.enc2(x)  # [batch_size, 64, 96, 96]
        x = self.pool(enc2)  # [batch_size, 64, 48, 48]

        enc3 = self.enc3(x)  # [batch_size, 128, 48, 48]
        x = self.pool(enc3)  # [batch_size, 128, 24, 24]

        enc4 = self.enc4(x)  # [batch_size, 256, 24, 24]

        # enhance last U-Net encoder feature with MiDaS features using FSCN
        x = self.f1(enc4, dpt_features)  # [batch_size, 256, 24, 24]
        x = nn.functional.interpolate(
            x, size=enc3.shape[2:], mode="bilinear", align_corners=True
        )

        # Feed enhanced feature into Decoder block with skip connection
        x = torch.cat([x, enc3], dim=1)
        x = self.dec4(x)  # [batch_size, 128, 48, 48]

        # enhance previous decoder feature with MiDaS features using FSCN
        x = self.f2(x, dpt_features)  # [batch_size, 128, 48, 48]
        x = nn.functional.interpolate(
            x, size=enc2.shape[2:], mode="bilinear", align_corners=True
        )

        # Feed enhanced feature into Decoder block with skip connection
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)  # [batch_size, 64, 96, 96]

        # enhance previous decoder feature with MiDaS features using FSCN
        x = self.f3(x, dpt_features)  # [batch_size, 64, 96, 96]
        x = nn.functional.interpolate(
            x, size=enc1.shape[2:], mode="bilinear", align_corners=True
        )

        # Feed enhanced feature into Decoder block with skip connection
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)  # [batch_size, 32, 192, 192]

        # enhance previous decoder feature with MiDaS features using FSCN
        x = self.f4(x, dpt_features)  # [batch_size, 32, 192, 192]
        x = self.dec1(x)  # [batch_size, 32, 192, 192]
        x = self.final(x)  # [batch_size, 1, 192, 192]

        # Output non-negative depth values
        x = torch.sigmoid(x) * 10

        return x

# ### Helper functions

def target_transform(depth):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=INPUT_SIZE,
        mode="bilinear",
        align_corners=True,
    ).squeeze()

    # Add channel dimension to match model output
    depth = depth.unsqueeze(0)
    return depth

def main():

    # Create output directories
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)

    # Define transforms
    print("Defining transforms...")
    train_transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create training dataset with ground truth
    print("Creating datasets...")
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file,
        transform=train_transform,
        target_transform=target_transform,
        has_gt=True,
    )

    # Create test dataset without ground truth
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        transform=test_transform,
        has_gt=False,  # Test set has no ground truth
    )

    # Split training dataset into train and validation
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)  # 85% for training
    val_size = total_size - train_size  # 15% for validation

    # Set a fixed random seed for reproducibility
    torch.manual_seed(0)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

    # Create data loaders with memory optimizations
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(
        f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )

    # Clear CUDA cache before model initialization
    torch.cuda.empty_cache()

    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
     
    model = EnhancedUNet(INPUT_SIZE)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(
            f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    # Define loss function and optimizer
    criterion = SILogLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Train the model
    print("Starting training...")
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, results_dir, in_epoch_validation=True,
    )

    # Evaluate the model on validation set
    print("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE)

    # Print metrics
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save metrics to file
    with open(os.path.join(results_dir, "validation_metrics.txt"), "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    # Generate predictions for the test set
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE)

    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")

main()
