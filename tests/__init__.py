"""Test package initialization."""


def test_version_import():
    """Test that version can be imported."""
    from fake_review_detection import __version__

    assert __version__ == "0.1.0"
