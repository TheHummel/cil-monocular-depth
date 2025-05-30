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
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (426, 560)
NUM_WORKERS = 2
PIN_MEMORY = True

# # Model 

# UNet Block
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

# Fusion block to combine MiDaS Encoder features with UNet features
class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.b = UNetBlock(in_channels, out_channels)

    def forward(self, midas_feature, unet_feature):
        midas_feature = nn.functional.interpolate(
            midas_feature, size=unet_feature.shape[2:], mode="bilinear", align_corners=True
        )
        out = torch.cat([unet_feature, midas_feature], dim=1)
        out = self.b(out)
        return unet_feature + out 

# The final model

class EnhancedUNet(nn.Module):
    def __init__(self):
        super(EnhancedUNet, self).__init__()
        
        # get pretrained encoder part of midas
        self.midas = DPTForDepthEstimation.from_pretrained("Intel/dpt-swinv2-base-384")

        # disable gradients
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
        
        # Fusion blocks
        self.f1 = FusionBlock(256 + 128, 256)
        self.f2 = FusionBlock(128 + 128, 128)
        self.f3 = FusionBlock(64 + 128, 64)
        self.f4 = FusionBlock(32 + 128, 32)

        # Final layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

        # projections to reduce dimensionality for fewer parameter count
        self.midas_feature_proj1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.midas_feature_proj2 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.midas_feature_proj3 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.midas_feature_proj4 = nn.Conv2d(1024, 128, kernel_size=1, padding=0)
    

    def forward(self, x):
        # compute features from midas encoder
        with torch.no_grad():
            # return list of 4 tensors of shapes:
            #   (batch_size, 128, 96, 96)
            #   (batch_size, 256, 48, 48)
            #   (batch_size, 512, 24, 24)
            #   (batch_size, 1024, 12, 12)
    
            midas_features = self.midas.backbone(x).feature_maps
        
        # project the 4 midas encoder fetures to lower channel dimensions
        mf1 = self.midas_feature_proj1(midas_features[0])
        mf2 = self.midas_feature_proj2(midas_features[1])
        mf3 = self.midas_feature_proj3(midas_features[2])
        mf4 = self.midas_feature_proj4(midas_features[3])

        # UNet  Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1) # (batch_size, 32, 192, 192)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2) # (batch_size, 64, 96, 96)
        
        enc3 = self.enc3(x) 
        x = self.pool(enc3) # (batch_size, 128, 48, 48)

        enc4 = self.enc4(x) # (batch_size, 256, 48, 48)
        
        # fuse unet encoder state  with midas encoder state
        x = self.f1(mf4, enc4) # output shape (batch_size, 256, 48, 48)

        # Decoder with skip connections
        x = nn.functional.interpolate(
            x, size=enc3.shape[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x, enc3], dim=1)
        x = self.dec4(x)

        # fuse decoder output with midas encoder state
        x = self.f2(mf3, x)
        x = nn.functional.interpolate(
            x, size=enc2.shape[2:], mode="bilinear", align_corners=True
        )

        # skip connection with UNet encoder
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)

        # fuse decoder output with midas encoder state
        x = self.f3(mf2, x)
        x = nn.functional.interpolate(
            x, size=enc1.shape[2:], mode="bilinear", align_corners=True
        )

        # skip connection with UNet encoder
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)

        # fuse decoder output with midas encoder state
        x = self.f4(mf1, x)
        
        x = self.dec1(x)
        x = self.final(x)

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
     
    model = EnhancedUNet()
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(
            f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    # Define loss function and optimizer
    criterion = scale_invariant_loss
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
