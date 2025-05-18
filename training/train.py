import os
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.helpers import print_tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    results_dir,
    in_epoch_validation=False,
):
    """Train the model with optional validation every 10% of training and save the best based on epoch-level validation metrics"""
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []
    model_saved = False

    # Calculate the number of batches for 10% of training if in_epoch_validation is enabled
    total_batches = len(train_loader)
    val_freq = 10
    val_interval = (
        max(1, total_batches // val_freq) if in_epoch_validation else total_batches
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        batch_count = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            if batch is None:
                print_tqdm(
                    f"Warning: Skipped training batch {batch_idx+1}/{total_batches}"
                )
                continue

            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).predicted_depth
            if outputs.dim() == 3:
                outputs = outputs.unsqueeze(1)
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            loss = criterion(outputs, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print_tqdm(
                    f"Warning: Invalid training loss at batch {batch_idx+1}: {loss.item()}"
                )
                continue

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            batch_count += 1

            if in_epoch_validation and (
                (batch_idx + 1) % val_interval == 0
                and (total_batches - (batch_idx + 1)) > 5
            ):
                # in-batch validation
                model.eval()
                val_loss = 0.0
                val_samples = 0

                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(
                        tqdm(val_loader, desc="In-batch Validation")
                    ):
                        if val_batch is None:
                            continue
                        val_inputs, val_targets, _ = val_batch
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(
                            device
                        )

                        val_outputs = model(val_inputs).predicted_depth
                        if val_outputs.dim() == 3:
                            val_outputs = val_outputs.unsqueeze(1)
                        val_outputs = nn.functional.interpolate(
                            val_outputs,
                            size=val_targets.shape[-2:],
                            mode="bilinear",
                            align_corners=True,
                        )

                        val_loss_batch = criterion(val_outputs, val_targets)

                        if torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch):
                            print_tqdm(
                                f"Warning: Invalid in-epoch validation loss at batch {val_batch_idx+1}: {val_loss_batch.item()}"
                            )
                            continue

                        val_loss += val_loss_batch.item() * val_inputs.size(0)
                        val_samples += val_inputs.size(0)

                if val_samples == 0:
                    print_tqdm(
                        f"Warning: No valid validation samples at {batch_idx+1}/{total_batches}"
                    )
                    model.train()
                    continue

                val_loss /= val_samples
                percentage = (batch_idx + 1) / total_batches * 100
                print_tqdm(
                    f"Validation at {percentage:.1f}% of epoch {epoch+1}: Validation Loss: {val_loss:.4f}"
                )

                with open(
                    os.path.join(results_dir, "in_epoch_val_losses.txt"), "a"
                ) as f:
                    f.write(f"Epoch {epoch+1}, {percentage:.1f}%: {val_loss:.4f}\n")

                model.train()

                # TODO: create checkpoints of the model

        # Compute average training loss for the epoch
        if train_samples == 0:
            print(f"Error: No valid training samples in epoch {epoch+1}")
            break
        train_loss /= train_samples
        train_losses.append(train_loss)

        # Epoch-level validation phase
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc="Epoch Validation")
            ):
                if batch is None:
                    continue
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs).predicted_depth
                if outputs.dim() == 3:
                    outputs = outputs.unsqueeze(1)
                outputs = nn.functional.interpolate(
                    outputs,
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

                loss = criterion(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"Warning: Invalid epoch-level validation loss at batch {batch_idx+1}: {loss.item()}"
                    )
                    continue

                val_loss += loss * inputs.size(0)
                val_samples += inputs.size(0)

        if val_samples == 0:
            print(f"Error: No valid validation samples for epoch {epoch+1}")
            break

        val_loss = val_loss / val_samples
        val_losses.append(val_loss.item())

        print(
            f"Train Loss: {train_loss:.4f}, Epoch Validation Loss: {val_loss.item():.4f}"
        )

        # Save the best model based on epoch-level validation loss
        if val_loss < best_val_loss and not (
            torch.isnan(val_loss) or torch.isinf(val_loss)
        ):
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
            try:
                torch.save(
                    model.state_dict(), os.path.join(results_dir, "best_model.pth")
                )
                model_saved = True
                print(
                    f"New best model saved at epoch {epoch+1} with epoch validation loss: {val_loss.item():.4f}"
                )
            except Exception as e:
                print(f"Error saving model: {e}")

    if not model_saved:
        print("Warning: No model was saved during training. Saving final model state.")
        try:
            torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pth"))
        except Exception as e:
            print(f"Error saving final model: {e}")
        return model

    print(
        f"\nBest model was from epoch {best_epoch} with epoch validation loss: {best_val_loss:.4f}"
    )

    # Load the best model
    try:
        model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    except FileNotFoundError:
        print("Error: best_model.pth not found. Loading final model instead.")
        model.load_state_dict(torch.load(os.path.join(results_dir, "final_model.pth")))

    return model
