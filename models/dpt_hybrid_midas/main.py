import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import DPTForDepthEstimation

from data.dataset import DepthDataset
from utils.helpers import (
    ensure_dir,
    custom_collate_fn
)
from training.train import train_model
from training.loss import SILogLoss
from inference.evaluate import evaluate_model, generate_test_predictions
from paths import *
from models.dpt_hybrid_midas.config import *

from models.dpt_hybrid_midas.channel_attention.CustomFeatureFusionLayer import CustomFeatureFusionLayer
from models.dpt_hybrid_midas.fscn.CustomFSCNFusionStage import CustomFSCNFusionStage
from models.dpt_hybrid_midas.tscn.CustomTSCNFusionStage import CustomTSCNFusionStage

def main():
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)

    # Define transforms
    print("Defining transforms...")
    train_transform = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    print("Creating datasets...")
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file,
        input_size=INPUT_SIZE,
        transform=train_transform,
        has_gt=True,
    )
    print(
        f"Skipped files in training dataset: {train_full_dataset.get_skipped_count()}"
    )
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        input_size=INPUT_SIZE,
        transform=test_transform,
        has_gt=False,
    )
    print(f"Skipped files in test dataset: {test_dataset.get_skipped_count()}")

    # Split training dataset
    total_size = len(train_full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size]
    )

    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=custom_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=custom_collate_fn,
    )
    print(
        f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Load pretrained model
    print("Loading pretrained model...")
    
    model_name = "Intel/dpt-hybrid-midas"
    model = DPTForDepthEstimation.from_pretrained(model_name)

    # # update skip connections with custom attention
    # for i in range(4):
    #     model.neck.fusion_stage.layers[i] = CustomDPTFeatureFusionLayer(model.config)

    # # update skip connections with FSCN
    model.neck.fusion_stage = CustomFSCNFusionStage(model.config)

    # # update skip connections with TSCN
    #model.neck.fusion_stage = CustomTSCNFusionStage(model.config)

    print("MODEL:")
    print(model)

    # Freeze embeddings backbone to save compute
    for param in model.dpt.parameters():
        param.requires_grad = False

    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(
            f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        )

    # Define loss and optimizer
    criterion = SILogLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Print config
    print("CONFIG:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Device: {DEVICE}")

    # Finetune model
    print("Starting finetuning...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        DEVICE,
        results_dir,
        in_epoch_validation=True,
    )

    # Evaluate model
    print("Evaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, DEVICE, results_dir)
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    with open(os.path.join(results_dir, "validation_metrics.txt"), "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    # Generate test predictions
    print("Generating predictions for test set...")
    generate_test_predictions(model, test_loader, DEVICE, predictions_dir)

    print(f"Results saved to {results_dir}")
    print(f"All test depth map predictions saved to {predictions_dir}")


if __name__ == "__main__":
    main()
