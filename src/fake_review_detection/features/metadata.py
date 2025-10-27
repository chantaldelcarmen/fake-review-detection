"""Metadata feature extraction."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetadataFeatureExtractor:
    """Extract metadata features from reviews."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_features(
        self,
        timestamps: Optional[List[str]] = None,
        product_categories: Optional[List[str]] = None,
        reviewer_locations: Optional[List[str]] = None,
        review_lengths: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Extract metadata features.

        Args:
            timestamps: List of review timestamps
            product_categories: List of product categories
            reviewer_locations: List of reviewer locations
            review_lengths: List of review text lengths

        Returns:
            DataFrame containing metadata features
        """
        num_samples = (
            len(timestamps)
            if timestamps
            else len(product_categories)
            if product_categories
            else len(review_lengths)
            if review_lengths
            else 0
        )
        logger.info(f"Extracting metadata features from {num_samples} samples...")

        features = []
        for i in range(num_samples):
            feature_dict = {}

            # Temporal features
            if timestamps and i < len(timestamps):
                feature_dict.update(self._extract_temporal_features(timestamps[i]))

            # Review length
            if review_lengths and i < len(review_lengths):
                feature_dict["review_length"] = review_lengths[i]

            features.append(feature_dict)

        df = pd.DataFrame(features)

        # Fill missing values
        if df.empty or len(df.columns) == 0:
            df = pd.DataFrame(
                {
                    "day_of_week": [0] * num_samples,
                    "hour_of_day": [0] * num_samples,
                    "is_weekend": [0] * num_samples,
                    "review_length": [0] * num_samples,
                }
            )

        logger.info(f"Extracted {len(df.columns)} metadata features")
        return df

    def _extract_temporal_features(self, timestamp: str) -> Dict[str, float]:
        """
        Extract temporal features from timestamp.

        Args:
            timestamp: Timestamp string

        Returns:
            Dictionary of temporal features
        """
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            day_of_week = dt.weekday()
            hour_of_day = dt.hour
            is_weekend = 1 if day_of_week >= 5 else 0

            return {
                "day_of_week": day_of_week,
                "hour_of_day": hour_of_day,
                "is_weekend": is_weekend,
            }
        except Exception:
            return {
                "day_of_week": 0,
                "hour_of_day": 0,
                "is_weekend": 0,
            }

    def extract_category_features(
        self, categories: List[str], use_onehot: bool = False
    ) -> pd.DataFrame:
        """
        Extract category features.

        Args:
            categories: List of categories
            use_onehot: Whether to use one-hot encoding

        Returns:
            DataFrame with category features
        """
        logger.info(f"Extracting category features from {len(categories)} samples...")

        if use_onehot:
            # One-hot encode categories
            df = pd.DataFrame({"category": categories})
            df = pd.get_dummies(df, columns=["category"], prefix="category")
        else:
            # Label encode categories
            unique_categories = list(set(categories))
            category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
            df = pd.DataFrame(
                {"category_encoded": [category_to_idx.get(cat, 0) for cat in categories]}
            )

        logger.info(f"Extracted {len(df.columns)} category features")
        return df
