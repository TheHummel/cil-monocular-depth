import os
import torch
from tqdm import tqdm


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
    """Train the model and save the best based on validation metrics"""
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []

    total_batches = len(train_loader)
    validation_interval = (
        max(1, total_batches // 10) if in_epoch_validation else total_batches
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
                continue
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).predicted_depth
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
            batch_count += 1

            if in_epoch_validation and (
                (batch_idx + 1) % validation_interval == 0
                or (batch_idx + 1) == total_batches
            ):
                # In-epoch validation phase
                model.eval()
                val_loss = 0.0
                val_samples = 0

                with torch.no_grad():
                    for val_batch in val_loader:
                        if val_batch is None:
                            continue
                        val_inputs, val_targets, _ = val_batch
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(
                            device
                        )

                        # Forward pass
                        val_outputs = model(val_inputs).predicted_depth
                        val_loss_batch = criterion(val_outputs, val_targets)

                        val_loss += val_loss_batch.item() * val_inputs.size(0)
                        val_samples += val_inputs.size(0)

                val_loss /= val_samples or 1
                percentage = (batch_idx + 1) / total_batches * 100
                print(
                    f"Validation at {percentage:.1f}% of epoch {epoch+1}: Validation Loss: {val_loss:.4f}"
                )

                model.train()

        # Compute average training loss for the epoch
        train_loss /= train_samples or 1
        train_losses.append(train_loss)

        # Epoch-level validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs).predicted_depth
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

        val_loss /= val_samples or 1
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Epoch Validation Loss: {val_loss:.4f}")

        # Save the best model based on epoch-level validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            print(
                f"New best model saved at epoch {epoch+1} with epoch validation loss: {val_loss:.4f}"
            )

    print(
        f"\nBest model was from epoch {best_epoch} with epoch validation loss: {best_val_loss:.4f}"
    )

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))

    return model
