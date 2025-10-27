"""Script to run feature ablation study."""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import logging
from pathlib import Path
import pandas as pd

from fake_review_detection.data.preprocessing import load_data
from fake_review_detection.data.dataset import ReviewDataset
from fake_review_detection.models.roberta_classifier import RobertaForReviewClassification
from fake_review_detection.evaluation.metrics import Evaluator
from fake_review_detection.features.ablation import FeatureAblationStudy
from fake_review_detection.features.linguistic import LinguisticFeatureExtractor
from fake_review_detection.features.behavioral import BehavioralFeatureExtractor
from fake_review_detection.features.metadata import MetadataFeatureExtractor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Run feature ablation study.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting feature ablation study...")
    logger.info(f"Configuration:\n{cfg}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.model.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model.model_name)

    # Load test data
    logger.info("Loading test data...")
    test_texts, test_labels = load_data(
        cfg.data.test_file, text_column=cfg.data.text_column, label_column=cfg.data.label_column
    )

    # Extract features
    logger.info("Extracting features...")
    linguistic_extractor = LinguisticFeatureExtractor()
    behavioral_extractor = BehavioralFeatureExtractor()
    metadata_extractor = MetadataFeatureExtractor()

    linguistic_features = linguistic_extractor.extract_features(test_texts)
    behavioral_features = behavioral_extractor.extract_features(
        reviewer_ids=None, product_ids=None, ratings=None
    )
    metadata_features = metadata_extractor.extract_features(review_lengths=[len(t) for t in test_texts])

    # Combine all features
    all_features = pd.concat([linguistic_features, behavioral_features, metadata_features], axis=1)
    logger.info(f"Total features: {all_features.shape[1]}")

    # Create dataset with features
    test_dataset = ReviewDataset(
        test_texts, test_labels, tokenizer, cfg.data.max_length, features=all_features
    )

    # Create data loader
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )

    # Initialize model with additional features
    logger.info("Initializing model...")
    model = RobertaForReviewClassification(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        dropout_rate=cfg.model.dropout_rate,
        use_additional_features=True,
        num_additional_features=all_features.shape[1],
    )

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pt"
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}")

    # Initialize evaluator and ablation study
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(model, device=device)
    ablation_study = FeatureAblationStudy(model, evaluator, device=device)

    # Define feature groups
    num_linguistic = linguistic_features.shape[1]
    num_behavioral = behavioral_features.shape[1]
    num_metadata = metadata_features.shape[1]

    feature_groups = {
        "linguistic": list(range(0, num_linguistic)),
        "behavioral": list(range(num_linguistic, num_linguistic + num_behavioral)),
        "metadata": list(
            range(num_linguistic + num_behavioral, num_linguistic + num_behavioral + num_metadata)
        ),
    }

    # Run ablation study
    results = ablation_study.run_ablation(test_dataloader, feature_groups)

    # Save results
    results_path = Path(cfg.output_dir) / "ablation_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    logger.info(f"Saved ablation results to {results_path}")


if __name__ == "__main__":
    main()
