"""Dataset class for fake review detection."""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional
import pandas as pd


class ReviewDataset(Dataset):
    """Dataset for review text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        features: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of review texts
            labels: List of labels (0 for genuine, 1 for fake)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            features: Additional features (linguistic, behavioral, metadata)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = features

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing input tensors
        """
        text = self.texts[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

        # Add label if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        # Add additional features if available
        if self.features is not None:
            feature_values = self.features.iloc[idx].values
            item["features"] = torch.tensor(feature_values, dtype=torch.float)

        return item
