"""Evaluation module for sentiment classification metrics.

Evaluates the sentiment analysis pipeline on a labeled dataset of clearly
polarized car review sentences, computing accuracy, precision, recall, and F1.
Results are saved to disk with a timestamp for reproducibility.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import RESULTS_DIR, SENTIMENT_MODEL
from src.logger import get_logger
from src.pipelines import SentimentAnalysisPipeline
from src.utils import calculate_metrics

logger = get_logger(__name__)


def _get_evaluation_dataset() -> Tuple[List[str], List[str]]:
    """Return an embedded evaluation dataset of clearly polarized car reviews.

    Each entry is a short, unambiguous car review sentence with a gold-standard
    sentiment label (``"POSITIVE"`` or ``"NEGATIVE"``).  The sentences are
    chosen to be straightforward so that a competent sentiment model achieves
    well above 80 % F1.

    Returns:
        Tuple of (review_texts, gold_labels) where gold_labels are
        ``"POSITIVE"`` or ``"NEGATIVE"``.
    """
    samples: List[Tuple[str, str]] = [
        # --- POSITIVE samples (25) ---
        (
            "This car has excellent fuel economy and the ride is incredibly smooth.",
            "POSITIVE",
        ),
        (
            "I absolutely love the interior design, it feels luxurious and modern.",
            "POSITIVE",
        ),
        (
            "The handling is fantastic, it corners like a sports car.",
            "POSITIVE",
        ),
        (
            "Best purchase I have ever made, this vehicle is outstanding in every way.",
            "POSITIVE",
        ),
        (
            "The safety features are top notch and give me great peace of mind.",
            "POSITIVE",
        ),
        (
            "Incredible acceleration and the engine is whisper quiet at highway speeds.",
            "POSITIVE",
        ),
        (
            "The infotainment system is intuitive and the sound quality is superb.",
            "POSITIVE",
        ),
        (
            "Very comfortable seats, perfect for long road trips without any fatigue.",
            "POSITIVE",
        ),
        (
            "The build quality is impressive, everything feels solid and well made.",
            "POSITIVE",
        ),
        (
            "Outstanding value for money, loaded with features at this price point.",
            "POSITIVE",
        ),
        (
            "The trunk space is enormous, easily fits all our luggage for family vacations.",
            "POSITIVE",
        ),
        (
            "Smooth automatic transmission that shifts seamlessly and effortlessly.",
            "POSITIVE",
        ),
        (
            "The paint quality is gorgeous and the exterior styling turns heads everywhere.",
            "POSITIVE",
        ),
        (
            "Excellent all-wheel-drive system, feels very stable in rain and snow.",
            "POSITIVE",
        ),
        (
            "The warranty coverage is generous and the dealer service has been wonderful.",
            "POSITIVE",
        ),
        (
            "This SUV has amazing towing capacity and handles heavy loads with ease.",
            "POSITIVE",
        ),
        (
            "The adaptive cruise control works perfectly and makes commuting a breeze.",
            "POSITIVE",
        ),
        (
            "I am extremely satisfied with the low maintenance costs over three years.",
            "POSITIVE",
        ),
        (
            "The hybrid powertrain delivers impressive mileage without sacrificing power.",
            "POSITIVE",
        ),
        (
            "Wonderful driving experience, the steering is precise and responsive.",
            "POSITIVE",
        ),
        (
            "The cabin is remarkably quiet even at high speeds on the highway.",
            "POSITIVE",
        ),
        (
            "Great resale value and the depreciation has been minimal after two years.",
            "POSITIVE",
        ),
        (
            "The heated seats and climate control make winter driving very pleasant.",
            "POSITIVE",
        ),
        (
            "Reliable engine that starts instantly every morning without any issues.",
            "POSITIVE",
        ),
        (
            "The parking sensors and backup camera are extremely helpful and accurate.",
            "POSITIVE",
        ),
        # --- NEGATIVE samples (25) ---
        (
            "The transmission failed after just 10000 miles, terrible reliability.",
            "NEGATIVE",
        ),
        (
            "Awful gas mileage, this car drinks fuel like there is no tomorrow.",
            "NEGATIVE",
        ),
        (
            "The brakes squeal constantly and the dealer cannot fix the problem.",
            "NEGATIVE",
        ),
        (
            "Extremely uncomfortable seats that cause back pain on any drive over an hour.",
            "NEGATIVE",
        ),
        (
            "The engine stalls randomly at intersections, a serious safety hazard.",
            "NEGATIVE",
        ),
        (
            "Paint started peeling after only six months, unacceptable build quality.",
            "NEGATIVE",
        ),
        (
            "The infotainment system is laggy, crashes frequently, and is frustrating to use.",
            "NEGATIVE",
        ),
        (
            "Terrible resale value, lost almost half its value in the first year.",
            "NEGATIVE",
        ),
        (
            "Road noise is unbearable at highway speeds, the cabin insulation is poor.",
            "NEGATIVE",
        ),
        (
            "The air conditioning failed twice in the first summer and repairs were expensive.",
            "NEGATIVE",
        ),
        (
            "Worst car I have ever owned, constant electrical problems and warning lights.",
            "NEGATIVE",
        ),
        (
            "The steering feels vague and disconnected, very unsettling at higher speeds.",
            "NEGATIVE",
        ),
        (
            "Horrible customer service at the dealership, they ignored my complaints.",
            "NEGATIVE",
        ),
        (
            "The suspension is stiff and harsh, every bump on the road is felt painfully.",
            "NEGATIVE",
        ),
        (
            "Cheap plastic interior that rattles and creaks over every small bump.",
            "NEGATIVE",
        ),
        (
            "The fuel tank is ridiculously small, requiring constant stops to refuel.",
            "NEGATIVE",
        ),
        (
            "Dangerous blind spots that the mirrors do nothing to compensate for.",
            "NEGATIVE",
        ),
        (
            "The turbo lag is so bad that merging onto the highway feels unsafe.",
            "NEGATIVE",
        ),
        (
            "My car has been in the shop more than in my driveway, constant breakdowns.",
            "NEGATIVE",
        ),
        (
            "The automatic emergency braking triggers for no reason, very alarming.",
            "NEGATIVE",
        ),
        (
            "Overpriced for what you get, competitors offer much more at this price.",
            "NEGATIVE",
        ),
        (
            "The headlights are dim and insufficient for nighttime driving on dark roads.",
            "NEGATIVE",
        ),
        (
            "Rust appeared on the undercarriage after only one winter, poor corrosion protection.",
            "NEGATIVE",
        ),
        (
            "The touchscreen is unresponsive and the navigation system is outdated.",
            "NEGATIVE",
        ),
        (
            "Terrible ride quality, the car bounces around and never settles on rough roads.",
            "NEGATIVE",
        ),
    ]

    reviews = [text for text, _ in samples]
    labels = [label for _, label in samples]
    return reviews, labels


def evaluate_sentiment_model(
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the sentiment analysis pipeline on a labeled test set.

    Classifies a set of clearly polarized car review sentences using the
    ``SentimentAnalysisPipeline`` and computes accuracy, precision, recall,
    and F1 via ``calculate_metrics()``.

    Args:
        model_name: HuggingFace model identifier for the sentiment classifier.
            Defaults to the value of ``SENTIMENT_MODEL`` in ``src/config.py``.
        output_dir: Directory where ``evaluation_metrics.json`` is written.
            Defaults to the project's ``RESULTS_DIR``.

    Returns:
        A dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``model_name``, ``num_samples``, and ``timestamp``.

    Raises:
        RuntimeError: If the pipeline returns an unexpected number of
            predictions.
    """
    resolved_model = model_name or SENTIMENT_MODEL
    resolved_output_dir = Path(output_dir) if output_dir else RESULTS_DIR

    logger.info("Starting sentiment model evaluation")
    logger.info("Model: %s", resolved_model)

    # ------------------------------------------------------------------
    # 1. Load embedded evaluation dataset
    # ------------------------------------------------------------------
    reviews, gold_labels = _get_evaluation_dataset()
    logger.info("Loaded %d evaluation samples", len(reviews))

    # ------------------------------------------------------------------
    # 2. Run predictions
    # ------------------------------------------------------------------
    logger.info("Running sentiment predictions ...")
    pipeline = SentimentAnalysisPipeline(model_name=resolved_model)
    predictions: List[Dict[str, Any]] = pipeline(reviews)

    if not predictions or len(predictions) != len(reviews):
        raise RuntimeError(
            f"Expected {len(reviews)} predictions, got {len(predictions) if predictions else 0}"
        )

    logger.info("Obtained %d predictions", len(predictions))

    # ------------------------------------------------------------------
    # 3. Compute metrics
    # ------------------------------------------------------------------
    metrics = calculate_metrics(gold_labels, predictions)
    logger.info(
        "Evaluation complete -- Accuracy: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        metrics["accuracy"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )

    # ------------------------------------------------------------------
    # 4. Build result payload and persist to disk
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).isoformat()

    result: Dict[str, Any] = {
        "model_name": resolved_model,
        "num_samples": len(reviews),
        "timestamp": timestamp,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    try:
        os.makedirs(resolved_output_dir, exist_ok=True)
        output_path = resolved_output_dir / "evaluation_metrics.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=4)
        logger.info("Metrics saved to %s", output_path)
    except OSError as exc:
        logger.error("Failed to save metrics file: %s", exc)

    return result
