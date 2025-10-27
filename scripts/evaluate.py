"""Script to evaluate the trained model."""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import logging
from pathlib import Path

from fake_review_detection.data.preprocessing import load_data
from fake_review_detection.data.dataset import ReviewDataset
from fake_review_detection.models.roberta_classifier import RobertaForReviewClassification
from fake_review_detection.evaluation.metrics import Evaluator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Evaluate the trained fake review detection model.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting model evaluation...")
    logger.info(f"Configuration:\n{cfg}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model.model_name)

    # Load test data
    logger.info("Loading test data...")
    test_texts, test_labels = load_data(
        cfg.data.test_file, text_column=cfg.data.text_column, label_column=cfg.data.label_column
    )

    # Create dataset
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, cfg.data.max_length)

    # Create data loader
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
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

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pt"
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, using untrained model")

    # Initialize evaluator
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(model, device=device)

    # Evaluate
    metrics = evaluator.evaluate(test_dataloader)
    logger.info(f"Test metrics: {metrics}")

    # Get detailed report
    report = evaluator.get_classification_report(test_dataloader)
    logger.info(f"Classification Report:\n{report}")

    # Get confusion matrix
    cm = evaluator.get_confusion_matrix(test_dataloader)
    logger.info(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
