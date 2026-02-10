"""
Shared pytest fixtures for the car reviews NLP test suite.

Provides mock models, sample data, and temporary directories so that
tests can run without downloading real HuggingFace models.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is importable regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Sample review data
# ---------------------------------------------------------------------------

SAMPLE_REVIEWS = [
    "I really enjoyed driving this car. The handling was excellent and the fuel economy was great.",
    "Terrible experience with this vehicle. The engine failed after only 5000 miles.",
    "The 2023 Model X has a beautiful interior and advanced safety features, but it's overpriced.",
    "Average car for the price. Nothing special about the performance or comfort.",
    "Best sedan I have ever owned. Quiet cabin, smooth ride, and outstanding reliability.",
]

SAMPLE_LABELS = ["positive", "negative", "mixed", "neutral", "positive"]


@pytest.fixture()
def sample_reviews():
    """Return a list of fake car review strings."""
    return list(SAMPLE_REVIEWS)


@pytest.fixture()
def sample_labels():
    """Return labels corresponding to ``sample_reviews``."""
    return list(SAMPLE_LABELS)


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs (plots, exports, etc.)."""
    out = tmp_path / "results"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# HuggingFace model mocks
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sentiment_pipeline():
    """Return a mock that behaves like a HuggingFace sentiment-analysis pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.95}]
    mock_pipe.side_effect = lambda texts, **kw: (
        [{"label": "POSITIVE", "score": 0.95}] * len(texts)
        if isinstance(texts, list)
        else [{"label": "POSITIVE", "score": 0.95}]
    )
    return mock_pipe


@pytest.fixture()
def mock_translation_pipeline():
    """Return a mock that behaves like a HuggingFace translation pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"translation_text": "Texto traducido de ejemplo."}]
    return mock_pipe


@pytest.fixture()
def mock_summarization_pipeline():
    """Return a mock that behaves like a HuggingFace summarization pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"summary_text": "This is a short summary."}]
    return mock_pipe


@pytest.fixture()
def mock_qa_pipeline():
    """Return a mock that behaves like a HuggingFace question-answering pipeline."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"answer": "excellent handling", "score": 0.88, "start": 40, "end": 59}
    return mock_pipe


@pytest.fixture()
def mock_transformers_pipeline(
    mock_sentiment_pipeline,
    mock_translation_pipeline,
    mock_summarization_pipeline,
    mock_qa_pipeline,
):
    """Patch ``transformers.pipeline`` so it returns the appropriate mock
    based on the *task* argument, without downloading any real models."""
    task_map = {
        "sentiment-analysis": mock_sentiment_pipeline,
        "translation": mock_translation_pipeline,
        "summarization": mock_summarization_pipeline,
        "question-answering": mock_qa_pipeline,
    }

    def _factory(task=None, model=None, **kwargs):
        return task_map.get(task, MagicMock())

    with patch("transformers.pipeline", side_effect=_factory) as patched:
        yield patched


@pytest.fixture()
def mock_model_cache():
    """Patch the project-level model cache so nothing is loaded from disk."""
    mock_cache = MagicMock()
    mock_cache.cache = {}
    mock_cache.get.return_value = None
    with patch("src.model_cache.model_cache", mock_cache):
        yield mock_cache
