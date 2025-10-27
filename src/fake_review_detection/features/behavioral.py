"""Behavioral feature extraction."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BehavioralFeatureExtractor:
    """Extract behavioral features from review metadata."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_features(
        self,
        reviewer_ids: Optional[List[str]] = None,
        product_ids: Optional[List[str]] = None,
        ratings: Optional[List[float]] = None,
        timestamps: Optional[List[str]] = None,
        helpful_votes: Optional[List[int]] = None,
        verified_purchases: Optional[List[bool]] = None,
    ) -> pd.DataFrame:
        """
        Extract behavioral features from review metadata.

        Args:
            reviewer_ids: List of reviewer IDs
            product_ids: List of product IDs
            ratings: List of ratings
            timestamps: List of timestamps
            helpful_votes: List of helpful vote counts
            verified_purchases: List of verified purchase flags

        Returns:
            DataFrame containing behavioral features
        """
        num_samples = len(reviewer_ids) if reviewer_ids else 0
        logger.info(f"Extracting behavioral features from {num_samples} samples...")

        features = []
        for i in range(num_samples):
            feature_dict = {}

            # Rating features
            if ratings:
                feature_dict["rating"] = ratings[i] if i < len(ratings) else 0.0
                feature_dict["is_extreme_rating"] = (
                    1 if ratings[i] in [1.0, 5.0] else 0 if i < len(ratings) else 0
                )

            # Helpful votes
            if helpful_votes:
                feature_dict["helpful_votes"] = helpful_votes[i] if i < len(helpful_votes) else 0

            # Verified purchase
            if verified_purchases:
                feature_dict["is_verified_purchase"] = (
                    1 if verified_purchases[i] else 0 if i < len(verified_purchases) else 0
                )

            features.append(feature_dict)

        df = pd.DataFrame(features)

        # Fill missing values
        if df.empty:
            df = pd.DataFrame(
                {
                    "rating": [0.0] * num_samples,
                    "is_extreme_rating": [0] * num_samples,
                    "helpful_votes": [0] * num_samples,
                    "is_verified_purchase": [0] * num_samples,
                }
            )

        logger.info(f"Extracted {len(df.columns)} behavioral features")
        return df

    def aggregate_reviewer_features(self, df: pd.DataFrame, reviewer_column: str) -> pd.DataFrame:
        """
        Aggregate features per reviewer.

        Args:
            df: DataFrame with review data
            reviewer_column: Name of the reviewer ID column

        Returns:
            DataFrame with aggregated features per reviewer
        """
        logger.info("Aggregating reviewer-level features...")

        # Count reviews per reviewer
        review_counts = df.groupby(reviewer_column).size().reset_index(name="review_count")

        # Average rating per reviewer
        if "rating" in df.columns:
            avg_ratings = (
                df.groupby(reviewer_column)["rating"].mean().reset_index(name="avg_reviewer_rating")
            )
            review_counts = review_counts.merge(avg_ratings, on=reviewer_column)

        logger.info(f"Aggregated features for {len(review_counts)} reviewers")
        return review_counts
