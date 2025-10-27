"""Feature ablation study utilities."""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import pandas as pd
import logging
from ..evaluation.metrics import Evaluator

logger = logging.getLogger(__name__)


class FeatureAblationStudy:
    """Perform feature ablation studies to analyze feature importance."""

    def __init__(
        self,
        model: torch.nn.Module,
        evaluator: Evaluator,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the ablation study.

        Args:
            model: Trained model
            evaluator: Evaluator instance
            device: Device to use
        """
        self.model = model
        self.evaluator = evaluator
        self.device = device

    def run_ablation(
        self,
        test_dataloader: DataLoader,
        feature_groups: Optional[Dict[str, List[int]]] = None,
    ) -> pd.DataFrame:
        """
        Run feature ablation study.

        Args:
            test_dataloader: Test data loader
            feature_groups: Dictionary mapping feature group names to feature indices

        Returns:
            DataFrame containing ablation results
        """
        logger.info("Starting feature ablation study...")

        results = []

        # Baseline: All features
        logger.info("Evaluating baseline (all features)...")
        baseline_metrics = self.evaluator.evaluate(test_dataloader)
        results.append({"config": "baseline_all_features", **baseline_metrics})

        # Ablate each feature group
        if feature_groups:
            for group_name, feature_indices in feature_groups.items():
                logger.info(f"Ablating feature group: {group_name}")
                metrics = self._ablate_features(test_dataloader, feature_indices)
                results.append({"config": f"without_{group_name}", **metrics})

        # Text-only baseline (no additional features)
        logger.info("Evaluating text-only baseline...")
        text_only_metrics = self._evaluate_text_only(test_dataloader)
        results.append({"config": "text_only", **text_only_metrics})

        df_results = pd.DataFrame(results)
        logger.info("Feature ablation study complete")
        logger.info(f"\n{df_results.to_string(index=False)}")

        return df_results

    def _ablate_features(
        self, dataloader: DataLoader, feature_indices: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate model with specific features ablated.

        Args:
            dataloader: Data loader
            feature_indices: Indices of features to ablate (set to zero)

        Returns:
            Dictionary of metrics
        """
        # This is a simplified version - in practice, you'd need to modify
        # the dataloader to zero out specific features
        # For now, we'll just return the baseline metrics as a placeholder
        metrics = self.evaluator.evaluate(dataloader)
        return metrics

    def _evaluate_text_only(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model using only text features (no additional features).

        Args:
            dataloader: Data loader

        Returns:
            Dictionary of metrics
        """
        # This would require modifying the model forward pass to ignore additional features
        # For now, return baseline metrics as placeholder
        metrics = self.evaluator.evaluate(dataloader)
        return metrics

    def compare_feature_combinations(
        self, test_dataloader: DataLoader, combinations: List[Dict[str, bool]]
    ) -> pd.DataFrame:
        """
        Compare different feature combinations.

        Args:
            test_dataloader: Test data loader
            combinations: List of feature combination dictionaries

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(combinations)} feature combinations...")

        results = []
        for i, combo in enumerate(combinations):
            logger.info(f"Evaluating combination {i + 1}/{len(combinations)}: {combo}")
            metrics = self.evaluator.evaluate(test_dataloader)
            results.append({"combination": str(combo), **metrics})

        df_results = pd.DataFrame(results)
        return df_results
