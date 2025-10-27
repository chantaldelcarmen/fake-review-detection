"""Tests for feature extractors."""

import pytest
from fake_review_detection.features.linguistic import LinguisticFeatureExtractor
from fake_review_detection.features.behavioral import BehavioralFeatureExtractor
from fake_review_detection.features.metadata import MetadataFeatureExtractor


def test_linguistic_feature_extraction():
    """Test linguistic feature extraction."""
    extractor = LinguisticFeatureExtractor()
    texts = ["This is a great product! I love it.", "Not good at all."]

    features = extractor.extract_features(texts)

    assert len(features) == 2
    assert "num_words" in features.columns
    assert "sentiment_polarity" in features.columns
    assert features["num_words"].iloc[0] > 0


def test_linguistic_feature_empty_text():
    """Test linguistic features with empty text."""
    extractor = LinguisticFeatureExtractor()
    texts = [""]

    features = extractor.extract_features(texts)

    assert len(features) == 1
    assert features["num_words"].iloc[0] == 0


def test_behavioral_feature_extraction():
    """Test behavioral feature extraction."""
    extractor = BehavioralFeatureExtractor()

    reviewer_ids = ["user1", "user2"]
    ratings = [5.0, 1.0]
    helpful_votes = [10, 0]
    verified_purchases = [True, False]

    features = extractor.extract_features(
        reviewer_ids=reviewer_ids,
        ratings=ratings,
        helpful_votes=helpful_votes,
        verified_purchases=verified_purchases,
    )

    assert len(features) == 2
    assert "rating" in features.columns
    assert "helpful_votes" in features.columns


def test_metadata_feature_extraction():
    """Test metadata feature extraction."""
    extractor = MetadataFeatureExtractor()

    timestamps = ["2023-01-15T10:30:00Z", "2023-12-25T22:00:00Z"]
    review_lengths = [100, 250]

    features = extractor.extract_features(timestamps=timestamps, review_lengths=review_lengths)

    assert len(features) == 2
    assert "day_of_week" in features.columns
    assert "review_length" in features.columns
