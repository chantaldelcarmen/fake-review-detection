"""Script to train the fake review detection model."""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import logging
from pathlib import Path
import wandb

from fake_review_detection.data.preprocessing import load_data
from fake_review_detection.data.dataset import ReviewDataset
from fake_review_detection.models.roberta_classifier import RobertaForReviewClassification
from fake_review_detection.models.trainer import Trainer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Train the fake review detection model.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting model training...")
    logger.info(f"Configuration:\n{cfg}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)

    # Initialize wandb if enabled
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, config=dict(cfg))

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model.model_name)

    # Load data
    logger.info("Loading training data...")
    train_texts, train_labels = load_data(
        cfg.data.train_file, text_column=cfg.data.text_column, label_column=cfg.data.label_column
    )
    val_texts, val_labels = load_data(
        cfg.data.val_file, text_column=cfg.data.text_column, label_column=cfg.data.label_column
    )

    # Create datasets
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, cfg.data.max_length)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, cfg.data.max_length)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle_train,
        num_workers=cfg.data.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )

    # Initialize model
    logger.info("Initializing model...")
    model = RobertaForReviewClassification(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        dropout_rate=cfg.model.dropout_rate,
        use_additional_features=cfg.model.use_additional_features,
        num_additional_features=cfg.model.num_additional_features,
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        use_wandb=cfg.use_wandb,
    )

    # Train
    trainer.train(num_epochs=cfg.training.num_epochs, save_dir=str(checkpoint_dir))

    logger.info("Training complete!")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
