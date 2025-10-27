"""Tests for data preprocessing utilities."""

import pytest
from fake_review_detection.data.preprocessing import TextPreprocessor, split_data


def test_text_preprocessor_lowercase():
    """Test text preprocessing with lowercase."""
    preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
    texts = ["This is a TEST", "Another EXAMPLE"]
    cleaned = preprocessor.preprocess(texts)

    assert cleaned[0] == "this is a test"
    assert cleaned[1] == "another example"


def test_text_preprocessor_remove_urls():
    """Test URL removal."""
    preprocessor = TextPreprocessor(lowercase=False, remove_urls=True)
    texts = ["Check this https://example.com for info", "Visit www.test.com"]
    cleaned = preprocessor.preprocess(texts)

    assert "https://example.com" not in cleaned[0]
    assert "www.test.com" not in cleaned[1]


def test_text_preprocessor_empty_text():
    """Test preprocessing with empty text."""
    preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
    cleaned = preprocessor.clean_text("")

    assert cleaned == ""


def test_split_data():
    """Test data splitting."""
    texts = [f"text_{i}" for i in range(100)]
    labels = [0] * 50 + [1] * 50

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        texts, labels, train_ratio=0.8, val_ratio=0.1
    )

    assert len(train_texts) == 80
    assert len(val_texts) == 10
    assert len(test_texts) == 10
    assert len(train_labels) == 80
    assert len(val_labels) == 10
    assert len(test_labels) == 10
