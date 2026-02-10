"""
Enhanced data processing module with multi-format support and validation.

This module provides robust data loading, validation, preprocessing,
and export capabilities for various file formats.
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import csv
from datetime import datetime

from src.logger import get_logger
from src.config import EXPORT_FORMATS, RESULTS_DIR
from src.error_handler import DataProcessingError, robust_operation

logger = get_logger(__name__)


class EnhancedDataProcessor:
    """Enhanced data processor with multi-format support and validation."""

    def __init__(self):
        self.supported_input_formats = [".csv", ".json", ".xlsx", ".tsv", ".txt"]
        self.supported_output_formats = ["json", "csv", "excel", "txt"]

    @robust_operation(fallback_value=([], []), context="data_loading")
    def load_data(self, file_path: Union[str, Path], **kwargs) -> Tuple[List[str], List[str]]:
        """
        Load car review data from various file formats.

        Args:
            file_path: Path to the data file
            **kwargs: Additional parameters for specific loaders

        Returns:
            Tuple of (reviews, labels)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataProcessingError(
                f"Data file not found: {file_path}",
                suggestions=[
                    "Check file path spelling",
                    "Ensure file exists in the specified location",
                    "Use absolute path if relative path fails",
                ],
            )

        logger.info(f"Loading data from {file_path}")

        # Determine file format and load accordingly
        file_ext = file_path.suffix.lower()

        if file_ext == ".csv":
            return self._load_csv(file_path, **kwargs)
        elif file_ext == ".json":
            return self._load_json(file_path, **kwargs)
        elif file_ext == ".xlsx":
            return self._load_excel(file_path, **kwargs)
        elif file_ext == ".tsv":
            return self._load_tsv(file_path, **kwargs)
        elif file_ext == ".txt":
            return self._load_txt(file_path, **kwargs)
        else:
            raise DataProcessingError(
                f"Unsupported file format: {file_ext}",
                suggestions=[
                    f"Supported formats: {', '.join(self.supported_input_formats)}",
                    "Convert your data to a supported format",
                    "Contact support for additional format support",
                ],
            )

    def _load_csv(self, file_path: Path, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from CSV file."""
        try:
            # Auto-detect delimiter
            delimiter = kwargs.pop("delimiter", None)
            if not delimiter:
                delimiter = self._detect_csv_delimiter(file_path)

            df = pd.read_csv(file_path, delimiter=delimiter, **kwargs)
            return self._extract_reviews_and_labels(df, file_path)

        except Exception as e:
            raise DataProcessingError(
                f"Failed to load CSV file: {e}",
                suggestions=[
                    "Check if file is properly formatted",
                    "Verify column names (expecting 'Review' and 'Class')",
                    "Try specifying delimiter explicitly",
                    "Check for special characters or encoding issues",
                ],
            ) from e

    def _load_json(self, file_path: Path, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                # List of objects format
                reviews = [item.get("review", item.get("text", "")) for item in data]
                labels = [
                    item.get("label", item.get("class", item.get("sentiment", ""))) for item in data
                ]
            elif isinstance(data, dict):
                # Dictionary format
                reviews = data.get("reviews", data.get("texts", []))
                labels = data.get("labels", data.get("classes", data.get("sentiments", [])))
            else:
                raise ValueError("Unsupported JSON structure")

            return self._validate_data(reviews, labels, file_path)

        except Exception as e:
            raise DataProcessingError(
                f"Failed to load JSON file: {e}",
                suggestions=[
                    "Check JSON syntax and structure",
                    "Ensure proper encoding (UTF-8)",
                    "Verify expected fields are present",
                    "Use online JSON validator to check format",
                ],
            ) from e

    def _load_excel(self, file_path: Path, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from Excel file."""
        try:
            sheet_name = kwargs.get("sheet_name", 0)  # Default to first sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            return self._extract_reviews_and_labels(df, file_path)

        except Exception as e:
            raise DataProcessingError(
                f"Failed to load Excel file: {e}",
                suggestions=[
                    "Ensure Excel file is not corrupted",
                    "Check if specified sheet exists",
                    "Verify column names match expected format",
                    "Install openpyxl: pip install openpyxl",
                ],
            ) from e

    def _load_tsv(self, file_path: Path, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from TSV file."""
        kwargs["delimiter"] = "\t"
        return self._load_csv(file_path, **kwargs)

    def _load_txt(self, file_path: Path, **kwargs) -> Tuple[List[str], List[str]]:
        """Load data from text file (one review per line)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            reviews = [line.strip() for line in lines if line.strip()]
            labels = ["UNKNOWN"] * len(reviews)  # Default label for text files

            return self._validate_data(reviews, labels, file_path)

        except Exception as e:
            raise DataProcessingError(
                f"Failed to load text file: {e}",
                suggestions=[
                    "Check file encoding (should be UTF-8)",
                    "Ensure file is readable",
                    "Verify file contains text data",
                ],
            ) from e

    def _detect_csv_delimiter(self, file_path: Path) -> str:
        """Auto-detect CSV delimiter."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sample = f.read(1024)

            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            logger.info(f"Detected CSV delimiter: '{delimiter}'")
            return delimiter

        except Exception:
            logger.warning("Failed to detect delimiter, using ';' as default")
            return ";"

    def _extract_reviews_and_labels(
        self, df: pd.DataFrame, file_path: Path
    ) -> Tuple[List[str], List[str]]:
        """Extract reviews and labels from DataFrame."""
        # Try different possible column names
        review_columns = ["Review", "review", "text", "Text", "comment", "Comment"]
        label_columns = ["Class", "class", "label", "Label", "sentiment", "Sentiment"]

        review_col = None
        label_col = None

        # Find review column
        for col in review_columns:
            if col in df.columns:
                review_col = col
                break

        # Find label column
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break

        if not review_col:
            raise DataProcessingError(
                f"No review column found in {file_path}",
                suggestions=[
                    f"Expected column names: {', '.join(review_columns)}",
                    "Check column headers in your file",
                    "Rename columns to match expected format",
                ],
            )

        reviews = df[review_col].fillna("").astype(str).tolist()

        if label_col:
            labels = df[label_col].fillna("UNKNOWN").astype(str).tolist()
        else:
            logger.warning(f"No label column found in {file_path}, using 'UNKNOWN' as default")
            labels = ["UNKNOWN"] * len(reviews)

        return self._validate_data(reviews, labels, file_path)

    def _validate_data(
        self, reviews: List[str], labels: List[str], file_path: Path
    ) -> Tuple[List[str], List[str]]:
        """Validate loaded data."""
        if not reviews:
            raise DataProcessingError(
                f"No reviews found in {file_path}",
                suggestions=[
                    "Check if file contains data",
                    "Verify file format is correct",
                    "Ensure reviews are in the expected column",
                ],
            )

        if len(reviews) != len(labels):
            logger.warning(f"Mismatch in reviews ({len(reviews)}) and labels ({len(labels)}) count")
            # Pad labels if needed
            if len(labels) < len(reviews):
                labels.extend(["UNKNOWN"] * (len(reviews) - len(labels)))
            else:
                labels = labels[: len(reviews)]

        # Filter out empty reviews
        valid_data = [(r, l) for r, l in zip(reviews, labels) if r.strip()]
        if len(valid_data) < len(reviews):
            logger.warning(f"Filtered out {len(reviews) - len(valid_data)} empty reviews")

        reviews, labels = zip(*valid_data) if valid_data else ([], [])

        logger.info(f"Successfully loaded {len(reviews)} reviews from {file_path}")
        return list(reviews), list(labels)

    def preprocess_reviews(
        self,
        reviews: List[str],
        min_length: int = 10,
        max_length: int = 5000,
        remove_duplicates: bool = True,
    ) -> List[str]:
        """
        Preprocess reviews with filtering and cleaning.

        Args:
            reviews: List of review texts
            min_length: Minimum review length
            max_length: Maximum review length
            remove_duplicates: Whether to remove duplicate reviews

        Returns:
            Preprocessed reviews
        """
        logger.info(f"Preprocessing {len(reviews)} reviews")

        # Basic cleaning
        processed = []
        for review in reviews:
            # Strip whitespace and normalize
            cleaned = review.strip()

            # Filter by length
            if len(cleaned) < min_length or len(cleaned) > max_length:
                continue

            processed.append(cleaned)

        logger.info(f"After length filtering: {len(processed)} reviews")

        # Remove duplicates if requested
        if remove_duplicates:
            original_count = len(processed)
            processed = list(dict.fromkeys(processed))  # Preserves order
            logger.info(f"Removed {original_count - len(processed)} duplicates")

        return processed

    @robust_operation(fallback_value=None, context="data_export")
    def export_results(
        self, results: Dict[str, Any], filename: str = None, formats: List[str] = None
    ) -> Dict[str, str]:
        """
        Export analysis results in multiple formats.

        Args:
            results: Analysis results dictionary
            filename: Base filename (without extension)
            formats: List of export formats

        Returns:
            Dictionary mapping format to output file path
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"car_reviews_analysis_{timestamp}"

        if not formats:
            formats = EXPORT_FORMATS

        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_files = {}

        for format_type in formats:
            try:
                if format_type == "json":
                    file_path = self._export_json(results, filename)
                elif format_type == "csv":
                    file_path = self._export_csv(results, filename)
                elif format_type == "excel":
                    file_path = self._export_excel(results, filename)
                elif format_type == "txt":
                    file_path = self._export_txt(results, filename)
                else:
                    logger.warning(f"Unsupported export format: {format_type}")
                    continue

                output_files[format_type] = file_path
                logger.info(f"Exported {format_type.upper()} to {file_path}")

            except Exception as e:
                logger.error(f"Failed to export {format_type}: {e}")

        return output_files

    def _export_json(self, results: Dict[str, Any], filename: str) -> str:
        """Export results to JSON format."""
        file_path = RESULTS_DIR / f"{filename}.json"

        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        return str(file_path)

    def _export_csv(self, results: Dict[str, Any], filename: str) -> str:
        """Export results to CSV format."""
        file_path = RESULTS_DIR / f"{filename}.csv"

        # Flatten results for CSV export
        flattened_data = []

        for task_name, task_data in results.items():
            if isinstance(task_data, dict):
                if "predictions" in task_data:
                    for i, pred in enumerate(task_data["predictions"]):
                        row = {
                            "task": task_name,
                            "index": i,
                            "prediction": pred.get("label", ""),
                            "confidence": pred.get("score", ""),
                        }
                        flattened_data.append(row)
                elif "metrics" in task_data:
                    for metric_name, metric_value in task_data["metrics"].items():
                        row = {
                            "task": task_name,
                            "metric": metric_name,
                            "value": metric_value,
                        }
                        flattened_data.append(row)

        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_csv(file_path, index=False)
        else:
            # Create empty CSV with headers
            pd.DataFrame(columns=["task", "metric", "value"]).to_csv(file_path, index=False)

        return str(file_path)

    def _export_excel(self, results: Dict[str, Any], filename: str) -> str:
        """Export results to Excel format with multiple sheets."""
        file_path = RESULTS_DIR / f"{filename}.xlsx"

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = []
            for task_name, task_data in results.items():
                if isinstance(task_data, dict) and "metrics" in task_data:
                    metrics = task_data["metrics"]
                    row = {"Task": task_name}
                    row.update(metrics)
                    summary_data.append(row)

            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Individual task sheets
            for task_name, task_data in results.items():
                if isinstance(task_data, dict) and "predictions" in task_data:
                    predictions_df = pd.DataFrame(task_data["predictions"])
                    predictions_df.to_excel(
                        writer, sheet_name=task_name[:31], index=False
                    )  # Excel sheet name limit

        return str(file_path)

    def _export_txt(self, results: Dict[str, Any], filename: str) -> str:
        """Export results to text format."""
        file_path = RESULTS_DIR / f"{filename}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Car Reviews Analysis Results\n")
            f.write("=" * 40 + "\n\n")

            for task_name, task_data in results.items():
                f.write(f"Task: {task_name.title()}\n")
                f.write("-" * 20 + "\n")

                if isinstance(task_data, dict):
                    if "metrics" in task_data:
                        f.write("Metrics:\n")
                        for metric, value in task_data["metrics"].items():
                            f.write(f"  {metric}: {value}\n")

                    if "predictions" in task_data:
                        f.write(f"Predictions: {len(task_data['predictions'])} items\n")

                f.write("\n")

        return str(file_path)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        else:
            return obj


# Backward compatibility alias
DataProcessor = EnhancedDataProcessor

# Global data processor instance
data_processor = EnhancedDataProcessor()
