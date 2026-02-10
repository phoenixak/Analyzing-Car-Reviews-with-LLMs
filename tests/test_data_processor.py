"""Tests for the EnhancedDataProcessor in src.data_processor.

Tests cover load_data (CSV, JSON, TXT, error paths), preprocess_reviews,
and export_results.  All file I/O uses temporary directories; time.sleep
is patched so robust_operation retries do not slow down the suite.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_processor import EnhancedDataProcessor, DataProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_retries():
    """Disable exponential-backoff sleeps inside robust_operation."""
    with patch("time.sleep"):
        yield


@pytest.fixture()
def processor():
    """Return a fresh EnhancedDataProcessor."""
    return EnhancedDataProcessor()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Verify constructor and backward-compat alias."""

    def test_supported_input_formats(self, processor):
        """Processor should list CSV, JSON, XLSX, TSV, and TXT as inputs."""
        expected = {".csv", ".json", ".xlsx", ".tsv", ".txt"}
        assert expected.issubset(set(processor.supported_input_formats))

    def test_supported_output_formats(self, processor):
        """Processor should list json, csv, excel, txt as outputs."""
        expected = {"json", "csv", "excel", "txt"}
        assert expected.issubset(set(processor.supported_output_formats))

    def test_dataprocessor_alias(self):
        """DataProcessor should be an alias for EnhancedDataProcessor."""
        assert DataProcessor is EnhancedDataProcessor


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadDataCSV:
    """CSV loading through load_data."""

    def test_semicolon_delimited(self, processor, tmp_path):
        """Load a semicolon-delimited CSV with Review/Class columns."""
        path = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "Review": ["Great car.", "Bad car."],
                "Class": ["POSITIVE", "NEGATIVE"],
            }
        )
        df.to_csv(path, sep=";", index=False)

        reviews, labels = processor.load_data(path)

        assert reviews == ["Great car.", "Bad car."]
        assert labels == ["POSITIVE", "NEGATIVE"]

    def test_comma_delimited_lowercase_columns(self, processor, tmp_path):
        """Load a comma-delimited CSV with lowercase column names."""
        path = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "review": ["Smooth ride.", "Rough drive."],
                "label": ["positive", "negative"],
            }
        )
        df.to_csv(path, sep=",", index=False)

        reviews, labels = processor.load_data(path)

        assert len(reviews) == 2
        assert reviews[0] == "Smooth ride."

    def test_missing_label_column_gets_unknown(self, processor, tmp_path):
        """CSV without a recognised label column assigns UNKNOWN labels."""
        path = tmp_path / "nolabel.csv"
        # Use two columns so csv.Sniffer can detect ',' as the delimiter.
        # A single-column CSV has no delimiters for the sniffer to detect.
        df = pd.DataFrame({"Review": ["Nice ride."], "extra": [42]})
        df.to_csv(path, sep=",", index=False)

        reviews, labels = processor.load_data(path)

        assert len(reviews) == 1
        assert labels == ["UNKNOWN"]

    def test_missing_review_column_returns_fallback(self, processor, tmp_path):
        """CSV without a recognised review column triggers fallback."""
        path = tmp_path / "bad.csv"
        df = pd.DataFrame({"foo": ["a"], "bar": [1]})
        df.to_csv(path, sep=",", index=False)

        reviews, labels = processor.load_data(path)

        assert reviews == []
        assert labels == []

    def test_empty_reviews_filtered(self, processor, tmp_path):
        """Rows with blank review text are filtered out."""
        path = tmp_path / "blanks.csv"
        df = pd.DataFrame(
            {
                "Review": ["Good.", "", "  ", "Bad."],
                "Class": ["pos", "neg", "neu", "neg"],
            }
        )
        df.to_csv(path, sep=",", index=False)

        reviews, labels = processor.load_data(path)

        assert len(reviews) == 2
        assert "Good." in reviews
        assert "Bad." in reviews


class TestLoadDataJSON:
    """JSON loading through load_data."""

    def test_list_format(self, processor, tmp_path):
        """Load JSON that is a list of {review, label} objects."""
        path = tmp_path / "data.json"
        data = [
            {"review": "Excellent!", "label": "positive"},
            {"review": "Terrible.", "label": "negative"},
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

        reviews, labels = processor.load_data(path)

        assert reviews == ["Excellent!", "Terrible."]
        assert labels == ["positive", "negative"]

    def test_list_format_text_key(self, processor, tmp_path):
        """Load JSON list using 'text' key instead of 'review'."""
        path = tmp_path / "data.json"
        data = [{"text": "Great ride.", "sentiment": "positive"}]
        path.write_text(json.dumps(data), encoding="utf-8")

        reviews, labels = processor.load_data(path)

        assert reviews == ["Great ride."]
        assert labels == ["positive"]

    def test_dict_format(self, processor, tmp_path):
        """Load JSON that is a dict with 'reviews' and 'labels' keys."""
        path = tmp_path / "data.json"
        data = {
            "reviews": ["Nice.", "Bad."],
            "labels": ["pos", "neg"],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

        reviews, labels = processor.load_data(path)

        assert len(reviews) == 2
        assert reviews[0] == "Nice."


class TestLoadDataTXT:
    """TXT loading through load_data."""

    def test_one_review_per_line(self, processor, tmp_path):
        """Each non-empty line becomes a review with UNKNOWN label."""
        path = tmp_path / "data.txt"
        path.write_text("Great car.\nBad car.\n\n", encoding="utf-8")

        reviews, labels = processor.load_data(path)

        assert reviews == ["Great car.", "Bad car."]
        assert all(lbl == "UNKNOWN" for lbl in labels)


class TestLoadDataTSV:
    """TSV loading through load_data (delegates to CSV with tab delimiter)."""

    def test_tab_delimited(self, processor, tmp_path):
        """Load a tab-delimited file."""
        path = tmp_path / "data.tsv"
        df = pd.DataFrame(
            {
                "Review": ["Smooth.", "Rough."],
                "Class": ["pos", "neg"],
            }
        )
        df.to_csv(path, sep="\t", index=False)

        reviews, labels = processor.load_data(path)

        assert len(reviews) == 2


class TestLoadDataErrors:
    """Error / fallback paths in load_data."""

    def test_nonexistent_file_returns_fallback(self, processor):
        """Missing file causes robust_operation to return ([], [])."""
        reviews, labels = processor.load_data(Path("/no/such/file.csv"))

        assert reviews == []
        assert labels == []

    def test_unsupported_extension_returns_fallback(self, processor, tmp_path):
        """Unsupported extension causes robust_operation to return ([], [])."""
        path = tmp_path / "data.parquet"
        path.write_text("data", encoding="utf-8")

        reviews, labels = processor.load_data(path)

        assert reviews == []
        assert labels == []

    def test_corrupt_json_returns_fallback(self, processor, tmp_path):
        """Malformed JSON triggers fallback."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")

        reviews, labels = processor.load_data(path)

        assert reviews == []
        assert labels == []


# ---------------------------------------------------------------------------
# preprocess_reviews
# ---------------------------------------------------------------------------


class TestPreprocessReviews:
    """Tests for preprocess_reviews."""

    def test_strips_whitespace(self, processor):
        """Leading/trailing whitespace should be stripped."""
        reviews = ["  Valid review with enough length.  "]
        result = processor.preprocess_reviews(reviews, min_length=5)

        assert result == ["Valid review with enough length."]

    def test_min_length_filter(self, processor):
        """Reviews shorter than min_length are dropped."""
        reviews = ["Short", "This review is long enough to pass."]
        result = processor.preprocess_reviews(reviews, min_length=10)

        assert len(result) == 1
        assert result[0] == "This review is long enough to pass."

    def test_max_length_filter(self, processor):
        """Reviews longer than max_length are dropped."""
        reviews = ["ok" * 100, "Short."]
        result = processor.preprocess_reviews(reviews, min_length=1, max_length=50)

        assert len(result) == 1
        assert result[0] == "Short."

    def test_remove_duplicates_true(self, processor):
        """Duplicate reviews are removed when remove_duplicates=True."""
        reviews = [
            "Duplicate review text here.",
            "Duplicate review text here.",
            "A unique review stands alone.",
        ]
        result = processor.preprocess_reviews(reviews, min_length=5, remove_duplicates=True)

        assert len(result) == 2

    def test_remove_duplicates_false(self, processor):
        """Duplicates are kept when remove_duplicates=False."""
        reviews = [
            "Duplicate review text here.",
            "Duplicate review text here.",
        ]
        result = processor.preprocess_reviews(reviews, min_length=5, remove_duplicates=False)

        assert len(result) == 2

    def test_empty_input(self, processor):
        """Empty list returns empty list."""
        assert processor.preprocess_reviews([]) == []


# ---------------------------------------------------------------------------
# export_results
# ---------------------------------------------------------------------------


class TestExportResults:
    """Tests for export_results (patches RESULTS_DIR to tmp_path)."""

    def test_export_json(self, processor, tmp_path):
        """Exporting to JSON creates a readable JSON file."""
        with patch("src.data_processor.RESULTS_DIR", tmp_path):
            results = {
                "sentiment": {
                    "metrics": {"accuracy": 0.9},
                    "predictions": [{"label": "POSITIVE", "score": 0.95}],
                }
            }
            output = processor.export_results(results, filename="test_out", formats=["json"])

        assert output is not None
        assert "json" in output
        with open(output["json"], "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert "sentiment" in loaded

    def test_export_csv(self, processor, tmp_path):
        """Exporting to CSV creates a valid CSV file."""
        with patch("src.data_processor.RESULTS_DIR", tmp_path):
            results = {
                "sentiment": {
                    "predictions": [
                        {"label": "POSITIVE", "score": 0.9},
                        {"label": "NEGATIVE", "score": 0.8},
                    ]
                }
            }
            output = processor.export_results(results, filename="test_out", formats=["csv"])

        assert output is not None
        assert "csv" in output
        assert os.path.exists(output["csv"])

    def test_export_txt(self, processor, tmp_path):
        """Exporting to TXT creates a readable text file."""
        with patch("src.data_processor.RESULTS_DIR", tmp_path):
            results = {"sentiment": {"metrics": {"accuracy": 0.85}}}
            output = processor.export_results(results, filename="test_out", formats=["txt"])

        assert output is not None
        assert "txt" in output
        content = Path(output["txt"]).read_text(encoding="utf-8")
        assert "Sentiment" in content

    def test_auto_timestamp_filename(self, processor, tmp_path):
        """Omitting filename produces a timestamped file."""
        with patch("src.data_processor.RESULTS_DIR", tmp_path):
            results = {"task": {"metrics": {"f1": 0.7}}}
            output = processor.export_results(results, formats=["json"])

        assert output is not None
        assert "json" in output

    def test_unsupported_format_skipped(self, processor, tmp_path):
        """Unknown format names are silently skipped."""
        with patch("src.data_processor.RESULTS_DIR", tmp_path):
            results = {"task": {"metrics": {"f1": 0.7}}}
            output = processor.export_results(results, filename="test_out", formats=["bogus"])

        assert output is not None
        assert "bogus" not in output

    def test_numpy_values_serializable(self, processor, tmp_path):
        """numpy types should be serialised without error."""
        with patch("src.data_processor.RESULTS_DIR", tmp_path):
            results = {
                "task": {
                    "metrics": {"accuracy": np.float64(0.85)},
                }
            }
            output = processor.export_results(results, filename="test_out", formats=["json"])

        assert output is not None
        assert "json" in output
