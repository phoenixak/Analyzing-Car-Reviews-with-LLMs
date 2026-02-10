"""
Utility functions for the car reviews analysis project.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from evaluate import load
import json

from src.logger import get_logger
from src.config import DATASET_PATH, RESULTS_DIR

# Set up logger
logger = get_logger(__name__)


def load_data(file_path: str = str(DATASET_PATH)) -> Tuple[List[str], List[str]]:
    """
    Load car review data from a CSV file.

    Args:
        file_path: Path to the CSV file containing car reviews.

    Returns:
        Tuple containing:
            - List of review texts
            - List of sentiment labels
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, delimiter=";")
        reviews = df["Review"].tolist()
        real_labels = df["Class"].tolist()
        logger.info(f"Loaded {len(reviews)} reviews")
        return reviews, real_labels
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def calculate_metrics(
    real_labels: List[str], predicted_labels: List[Dict]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for sentiment analysis.

    Args:
        real_labels: List of true sentiment labels.
        predicted_labels: List of predicted sentiment labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info("Calculating evaluation metrics")
    try:
        accuracy = load("accuracy")
        f1 = load("f1")
        precision = load("precision")
        recall = load("recall")

        references = [1 if label == "POSITIVE" else 0 for label in real_labels]
        predictions = [
            1 if label["label"] == "POSITIVE" else 0 for label in predicted_labels
        ]

        accuracy_result = accuracy.compute(
            references=references, predictions=predictions
        )["accuracy"]
        f1_result = f1.compute(references=references, predictions=predictions)["f1"]
        precision_result = precision.compute(
            references=references, predictions=predictions
        )["precision"]
        recall_result = recall.compute(references=references, predictions=predictions)[
            "recall"
        ]

        metrics = {
            "accuracy": accuracy_result,
            "f1": f1_result,
            "precision": precision_result,
            "recall": recall_result,
        }

        logger.info(f"Metrics: {metrics}")
        logger.info(f"Accuracy: {accuracy_result:.4f}")
        logger.info(f"F1 score: {f1_result:.4f}")
        logger.info(f"Precision: {precision_result:.4f}")
        logger.info(f"Recall: {recall_result:.4f}")

        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


def calculate_bleu_score(translated_review: str, references: List[str]) -> float:
    """
    Calculate BLEU score for translation evaluation.

    Args:
        translated_review: Model-generated translation.
        references: List of reference translations.

    Returns:
        BLEU score.
    """
    logger.info("Calculating BLEU score")
    try:
        bleu = load("bleu")
        score = bleu.compute(predictions=[translated_review], references=[references])[
            "bleu"
        ]
        logger.info(f"BLEU score: {score}")
        return score
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        raise


def save_results(results: Dict[str, Any], filename: str) -> str:
    """
    Save analysis results to a JSON file.

    Args:
        results: Dictionary of results to save.
        filename: Name of the file to save results to.

    Returns:
        Path to the saved file.
    """
    logger.info(f"Saving results to {filename}")
    try:
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save results
        file_path = os.path.join(RESULTS_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Results saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


# NOTE: Visualization functions have been moved to src/visualization.py
# Import them from there to avoid code duplication
# from src.visualization import generate_wordcloud, plot_sentiment_distribution
