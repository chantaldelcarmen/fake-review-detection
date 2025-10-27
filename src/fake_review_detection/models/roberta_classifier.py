"""RoBERTa-based model for fake review detection."""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Optional


class RobertaForReviewClassification(nn.Module):
    """RoBERTa model for binary classification with optional feature fusion."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        use_additional_features: bool = False,
        num_additional_features: int = 0,
    ):
        """
        Initialize the model.

        Args:
            model_name: Name of the pretrained RoBERTa model
            num_labels: Number of output labels
            dropout_rate: Dropout rate
            use_additional_features: Whether to use additional features
            num_additional_features: Number of additional features
        """
        super().__init__()

        self.use_additional_features = use_additional_features
        self.num_additional_features = num_additional_features

        # Load pretrained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.config = self.roberta.config

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate classifier input size
        classifier_input_size = self.config.hidden_size
        if use_additional_features:
            classifier_input_size += num_additional_features

        # Classification head
        self.classifier = nn.Linear(classifier_input_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            features: Additional features (optional)
            labels: Labels for computing loss (optional)

        Returns:
            Dictionary containing logits and optionally loss
        """
        # Get RoBERTa outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Concatenate with additional features if provided
        if self.use_additional_features and features is not None:
            pooled_output = torch.cat([pooled_output, features], dim=1)

        # Get logits
        logits = self.classifier(pooled_output)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def freeze_encoder(self):
        """Freeze the RoBERTa encoder parameters."""
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the RoBERTa encoder parameters."""
        for param in self.roberta.parameters():
            param.requires_grad = True
