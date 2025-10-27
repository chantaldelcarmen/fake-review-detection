"""Model evaluation utilities."""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for the fake review detection model."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            device: Device to use for evaluation
        """
        self.model = model.to(device)
        self.device = device

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from the model.

        Args:
            dataloader: Data loader

        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                logits = outputs["logits"]

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
                if "labels" in batch:
                    all_labels.extend(batch["labels"].cpu().numpy())

        return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataloader: Data loader

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model...")

        predictions, true_labels, probs = self.predict(dataloader)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="binary"
        )

        # ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, probs)
        except ValueError:
            roc_auc = 0.0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def get_classification_report(self, dataloader: DataLoader) -> str:
        """
        Get a detailed classification report.

        Args:
            dataloader: Data loader

        Returns:
            Classification report as string
        """
        predictions, true_labels, _ = self.predict(dataloader)
        report = classification_report(
            true_labels, predictions, target_names=["Genuine", "Fake"], digits=4
        )
        return report

    def get_confusion_matrix(self, dataloader: DataLoader) -> np.ndarray:
        """
        Get the confusion matrix.

        Args:
            dataloader: Data loader

        Returns:
            Confusion matrix
        """
        predictions, true_labels, _ = self.predict(dataloader)
        cm = confusion_matrix(true_labels, predictions)
        return cm
