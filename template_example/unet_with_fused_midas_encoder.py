#!/usr/bin/env python
# coding: utf-8

# # Prep

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DPTForDepthEstimation

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
WEIGHT_DECAY = 1e-6
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (426, 560)
NUM_WORKERS = 2
PIN_MEMORY = True

# ### Helper functions

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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


# # Dataset

class DepthDataset(Dataset):
    def __init__(
        self, data_dir, list_file, transform=None, target_transform=None, has_gt=True
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt

        # Read file list
        with open(list_file, "r") as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f if line.strip()]
            else:
                # For test set without ground truth
                self.file_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)

    def __getitem__(self, idx):
        if self.has_gt:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])

            # Load RGB image
            rgb = Image.open(rgb_path).convert("RGB")
            
            #grayscale = rgb.convert("L")
            #grayscale = grayscale.filter(ImageFilter.SMOOTH)
            #edges = grayscale.filter(ImageFilter.FIND_EDGES)
            #edges = transforms.ToTensor()(edges)

            # Load depth map
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)

            # Apply transformations
            if self.transform:
                rgb = self.transform(rgb)

            if self.target_transform:
                depth = self.target_transform(depth)
            else:
                # Add channel dimension if not done by transform
                depth = depth.unsqueeze(0)
            
            ## augment input image with edge detection
            #rgb = torch.cat([rgb, edges], dim=0)

            return (
                rgb,
                depth,
                self.file_pairs[idx][0],
            )  # Return filename for saving predictions
        else:
            # For test set without ground truth
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(" ")[0])

            # Load RGB image
            rgb = Image.open(rgb_path).convert("RGB")
            
            #grayscale = rgb.convert("L")
            #grayscale = grayscale.filter(ImageFilter.SMOOTH)
            #edges = grayscale.filter(ImageFilter.FIND_EDGES)
            #edges = transforms.ToTensor()(edges)

            # Apply transformations
            if self.transform:
                rgb = self.transform(rgb)
            
            ## augment input image with edge detection
            #rgb = torch.cat([rgb, edges], dim=0)

            return rgb, self.file_list[idx]  # No depth, just return the filename


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
    def __init__(self, output_size):
        super(EnhancedUNet, self).__init__()

        self.output_size = output_size

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

        # Adjusted: Use only 2 DPT features for efficiency (layers 6 and 12, corresponding to H/8 and H/32)
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

        # Fuse U-Net feature with DPT features using FSCN
        #x = torch.flatten(x, start_dim=2).permute(0, 2, 1)
        x = self.f1(enc4, dpt_features)  # [batch_size, 256, 24, 24]
        #x = x.permute(0, 2, 1).reshape(enc4.shape)

        # Decoder with skip connections
        x = nn.functional.interpolate(
            x, size=enc3.shape[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x, enc3], dim=1)
        x = self.dec4(x)  # [batch_size, 128, 48, 48]

        x = self.f2(x, dpt_features)  # [batch_size, 128, 48, 48]
        x = nn.functional.interpolate(
            x, size=enc2.shape[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)  # [batch_size, 64, 96, 96]

        x = self.f3(x, dpt_features)  # [batch_size, 64, 96, 96]
        x = nn.functional.interpolate(
            x, size=enc1.shape[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x, enc1], dim=1)
        x = self.dec2(x)  # [batch_size, 32, 192, 192]

        x = self.f4(x, dpt_features)  # [batch_size, 32, 192, 192]
        x = self.dec1(x)  # [batch_size, 32, 192, 192]
        x = self.final(x)  # [batch_size, 1, 192, 192]

        # Output non-negative depth values
        x = torch.sigmoid(x) * 10

        x = nn.functional.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=True
        )
        return x


# # Training loop

class SILogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 1e-6).float()
        pred = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)
        diff_log = torch.log(pred) - torch.log(target)
        diff_log = diff_log * valid_mask
        count = torch.sum(valid_mask) + 1e-6
        log_mean = torch.sum(diff_log) / count
        squared_term = torch.sum(diff_log**2) / count
        mean_term = log_mean**2
        loss = torch.sqrt(squared_term + mean_term)
        return loss

def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    """Train the model and save the best based on validation metrics"""
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets, _ in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            print(
                f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}"
            )

    print(
        f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}"
    )

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    
    # save validation error per epoch in file for easier access
    with open(os.path.join(results_dir, 'per_epoch_validation_scores.txt'), 'w') as f:
        for i, score in zip(range(1, 1+len(val_losses)), val_losses):
            f.write(f"{i},{score:.4f}\n")

    return model


# # Model evaluation

def evaluate_model(model, val_loader, device):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()

    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0

    total_samples = 0
    target_shape = None

    with torch.no_grad():
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if target_shape is None:
                target_shape = targets.shape

            # Forward pass
            outputs = model(inputs)

            # Resize outputs to match target dimensions
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],  # Match height and width of targets
                mode="bilinear",
                align_corners=True,
            )

            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()

            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()

                EPSILON = 1e-6

                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue

                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]

                log_target = np.log(target_valid)

                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)

                # Calculate scale-invariant error
                diff = log_pred - log_target
                diff_mean = np.mean(diff)

                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))

            # Calculate thresholded accuracy
            max_ratio = torch.max(
                outputs / (targets + 1e-6), targets / (outputs + 1e-6)
            )
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()

            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i

                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()

                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (
                        input_np.max() - input_np.min() + 1e-6
                    )

                    # Create visualization
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(target_np, cmap="plasma")
                    plt.title("Ground Truth Depth")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(output_np, cmap="plasma")
                    plt.title("Predicted Depth")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    plt.close()

            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio

        # Clear CUDA cache
        torch.cuda.empty_cache()

    # Calculate final metrics using stored target shape
    total_pixels = (
        target_shape[1] * target_shape[2] * target_shape[3]
    )  # channels * height * width
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "siRMSE": sirmse,
        "REL": rel,
        "Delta1": delta1,
        "Delta2": delta2,
        "Delta3": delta3,
    }

    return metrics


# # Generate test predictions

def generate_test_predictions(model, test_loader, device):
    """Generate predictions for the test set without ground truth"""
    model.eval()

    # Ensure predictions directory exists
    ensure_dir(predictions_dir)

    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs = model(inputs)

            # Resize outputs to match original input dimensions (426x560)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode="bilinear",
                align_corners=True,
            )

            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(" ")[1]

                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)

            # Clean up memory
            del inputs, outputs

        # Clear cache after test predictions
        torch.cuda.empty_cache()


# # Putting it all together

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
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE
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

# Open a sample prediction from validation set
#Image.open(os.path.join(results_dir, "sample_0.png")
