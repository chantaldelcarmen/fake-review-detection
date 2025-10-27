"""Test configuration file."""

import pytest


@pytest.fixture
def sample_texts():
    """Sample review texts for testing."""
    return [
        "This product is amazing! I love it so much.",
        "Terrible quality, do not buy.",
        "Average product, nothing special.",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return [0, 1, 0]  # 0 = genuine, 1 = fake
