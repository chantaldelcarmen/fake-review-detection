"""Tests for the ReviewDataset class."""

import pytest
import torch
from transformers import RobertaTokenizer
from fake_review_detection.data.dataset import ReviewDataset


@pytest.fixture
def tokenizer():
    """Load RoBERTa tokenizer."""
    return RobertaTokenizer.from_pretrained("roberta-base")


def test_dataset_creation(tokenizer):
    """Test dataset creation."""
    texts = ["This is a test", "Another test"]
    labels = [0, 1]

    dataset = ReviewDataset(texts, labels, tokenizer, max_length=128)

    assert len(dataset) == 2


def test_dataset_getitem(tokenizer):
    """Test getting item from dataset."""
    texts = ["This is a test"]
    labels = [1]

    dataset = ReviewDataset(texts, labels, tokenizer, max_length=128)
    item = dataset[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)


def test_dataset_without_labels(tokenizer):
    """Test dataset without labels."""
    texts = ["This is a test"]

    dataset = ReviewDataset(texts, labels=None, tokenizer=tokenizer, max_length=128)
    item = dataset[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" not in item
