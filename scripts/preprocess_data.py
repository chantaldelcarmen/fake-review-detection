"""Script to preprocess raw data."""

import hydra
from omegaconf import DictConfig
import pandas as pd
import logging
from pathlib import Path

from fake_review_detection.data.preprocessing import TextPreprocessor, split_data

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Preprocess raw data and split into train/val/test sets.

    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting data preprocessing...")
    logger.info(f"Configuration:\n{cfg}")

    # Create output directory
    output_dir = Path(cfg.data_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    raw_data_path = Path(cfg.data_dir) / "raw" / "reviews.csv"
    if not raw_data_path.exists():
        logger.error(f"Raw data file not found: {raw_data_path}")
        logger.info(
            "Please place your raw data in data/raw/reviews.csv with 'text' and 'label' columns"
        )
        return

    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df)} samples from {raw_data_path}")

    # Preprocess text
    preprocessor = TextPreprocessor(lowercase=cfg.data.lowercase, remove_urls=cfg.data.remove_urls)
    df["text"] = preprocessor.preprocess(df[cfg.data.text_column].tolist())

    # Split data
    texts = df["text"].tolist()
    labels = df[cfg.data.label_column].tolist()

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        texts, labels, train_ratio=cfg.data.train_ratio, val_ratio=cfg.data.val_ratio
    )

    # Save splits
    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    val_df = pd.DataFrame({"text": val_texts, "label": val_labels})
    test_df = pd.DataFrame({"text": test_texts, "label": test_labels})

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Saved preprocessed data to {output_dir}")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


if __name__ == "__main__":
    main()
