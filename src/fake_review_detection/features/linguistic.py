"""Linguistic feature extraction."""

import pandas as pd
import numpy as np
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logger.warning("Downloading NLTK punkt tokenizer...")
    nltk.download("punkt", quiet=True)


class LinguisticFeatureExtractor:
    """Extract linguistic features from review text."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract linguistic features from texts.

        Args:
            texts: List of review texts

        Returns:
            DataFrame containing linguistic features
        """
        logger.info(f"Extracting linguistic features from {len(texts)} texts...")

        features = []
        for text in texts:
            features.append(self._extract_single(text))

        df = pd.DataFrame(features)
        logger.info(f"Extracted {len(df.columns)} linguistic features")
        return df

    def _extract_single(self, text: str) -> Dict[str, float]:
        """
        Extract features from a single text.

        Args:
            text: Review text

        Returns:
            Dictionary of features
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_features()

        # Tokenize
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)

        # Basic statistics
        num_words = len(words)
        num_sentences = len(sentences)
        num_chars = len(text)

        # Average lengths
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

        # Sentiment analysis
        try:
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
        except Exception:
            sentiment_polarity = 0.0
            sentiment_subjectivity = 0.0

        # Lexical diversity
        unique_words = len(set(words))
        lexical_diversity = unique_words / num_words if num_words > 0 else 0

        # Punctuation and capitalization
        num_exclamations = text.count("!")
        num_questions = text.count("?")
        num_uppercase = sum(1 for c in text if c.isupper())
        uppercase_ratio = num_uppercase / num_chars if num_chars > 0 else 0

        return {
            "num_words": num_words,
            "num_sentences": num_sentences,
            "num_chars": num_chars,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "sentiment_polarity": sentiment_polarity,
            "sentiment_subjectivity": sentiment_subjectivity,
            "lexical_diversity": lexical_diversity,
            "num_exclamations": num_exclamations,
            "num_questions": num_questions,
            "uppercase_ratio": uppercase_ratio,
        }

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            "num_words": 0,
            "num_sentences": 0,
            "num_chars": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "sentiment_polarity": 0,
            "sentiment_subjectivity": 0,
            "lexical_diversity": 0,
            "num_exclamations": 0,
            "num_questions": 0,
            "uppercase_ratio": 0,
        }
