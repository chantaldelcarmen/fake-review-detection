"""Data preprocessing utilities."""

import pandas as pd
import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocessor for review text data."""

    def __init__(self, lowercase: bool = True, remove_urls: bool = True):
        """
        Initialize the preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_urls: Whether to remove URLs
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls

    def clean_text(self, text: str) -> str:
        """
        Clean a single text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r"http\S+|www.\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        return text

    def preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        logger.info(f"Preprocessing {len(texts)} texts...")
        cleaned_texts = [self.clean_text(text) for text in texts]
        logger.info("Preprocessing complete.")
        return cleaned_texts


def load_data(
    filepath: str, text_column: str = "text", label_column: str = "label"
) -> Tuple[List[str], List[int]]:
    """
    Load data from a CSV file.

    Args:
        filepath: Path to the data file
        text_column: Name of the text column
        label_column: Name of the label column

    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    texts = df[text_column].tolist()
    labels = df[label_column].tolist() if label_column in df.columns else None

    logger.info(f"Loaded {len(texts)} samples")
    return texts, labels


def split_data(
    texts: List[str], labels: List[int], train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Split data into train, validation, and test sets.

    Args:
        texts: List of texts
        labels: List of labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data

    Returns:
        Tuple of (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split

    # First split: train and temp (val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, train_size=train_ratio, random_state=42, stratify=labels
    )

    # Second split: val and test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, train_size=val_ratio_adjusted, random_state=42, stratify=temp_labels
    )

    logger.info(f"Split data: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
