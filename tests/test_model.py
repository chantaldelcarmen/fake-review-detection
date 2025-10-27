"""Tests for the RoBERTa classifier model."""

import pytest
import torch
from fake_review_detection.models.roberta_classifier import RobertaForReviewClassification


def test_model_initialization():
    """Test model initialization."""
    model = RobertaForReviewClassification(
        model_name="roberta-base", num_labels=2, dropout_rate=0.1
    )

    assert model is not None
    assert model.classifier.out_features == 2


def test_model_forward():
    """Test forward pass."""
    model = RobertaForReviewClassification(
        model_name="roberta-base", num_labels=2, dropout_rate=0.1
    )

    batch_size = 2
    seq_length = 128

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.tensor([0, 1])

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, 2)
    assert outputs["loss"] is not None


def test_model_with_additional_features():
    """Test model with additional features."""
    model = RobertaForReviewClassification(
        model_name="roberta-base",
        num_labels=2,
        dropout_rate=0.1,
        use_additional_features=True,
        num_additional_features=10,
    )

    batch_size = 2
    seq_length = 128

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    features = torch.randn(batch_size, 10)
    labels = torch.tensor([0, 1])

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features, labels=labels)

    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, 2)


def test_model_freeze_unfreeze():
    """Test freezing and unfreezing encoder."""
    model = RobertaForReviewClassification(model_name="roberta-base", num_labels=2)

    # Freeze encoder
    model.freeze_encoder()
    for param in model.roberta.parameters():
        assert not param.requires_grad

    # Unfreeze encoder
    model.unfreeze_encoder()
    for param in model.roberta.parameters():
        assert param.requires_grad
