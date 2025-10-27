"""Training utilities for the model."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Optional
import wandb

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the fake review detection model."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            device: Device to use for training
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.use_wandb = use_wandb

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs["logits"], dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "acc": correct / total})

        metrics = {"train_loss": total_loss / len(self.train_dataloader), "train_acc": correct / total}

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary containing validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"]

                # Calculate metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs["logits"], dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        metrics = {"val_loss": total_loss / len(self.val_dataloader), "val_acc": correct / total}

        return metrics

    def train(self, num_epochs: int, save_dir: Optional[str] = None) -> None:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train metrics: {train_metrics}")

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                logger.info(f"Validation metrics: {val_metrics}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({**train_metrics, **val_metrics, "epoch": epoch + 1})

            # Save best model
            if save_dir and val_metrics and val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                checkpoint_path = f"{save_dir}/best_model.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

        logger.info("Training complete")
