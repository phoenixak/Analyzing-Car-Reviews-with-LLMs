#!/usr/bin/env python
"""Standalone script to evaluate the sentiment analysis model.

Usage::

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --output-dir results/eval
    python scripts/run_evaluation.py --model distilbert-base-uncased-finetuned-sst-2-english

The script prints a formatted report to stdout and persists the metrics as
JSON in the specified output directory (defaults to ``results/``).
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` is importable when the
# script is executed directly from the ``scripts/`` directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation import evaluate_sentiment_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the sentiment analysis pipeline on a labeled test set."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation_metrics.json (default: project results/ dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name for sentiment classification",
    )
    return parser.parse_args()


def _print_report(metrics: dict) -> None:
    """Print a human-readable evaluation report to stdout."""
    width = 52
    print()
    print("=" * width)
    print("  Sentiment Model Evaluation Report")
    print("=" * width)
    print(f"  Model      : {metrics['model_name']}")
    print(f"  Samples    : {metrics['num_samples']}")
    print(f"  Timestamp  : {metrics['timestamp']}")
    print("-" * width)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print("-" * width)
    f1 = metrics["f1"]
    if f1 >= 0.80:
        print(f"  Result     : PASS (F1 {f1:.2%} >= 80%)")
    else:
        print(f"  Result     : FAIL (F1 {f1:.2%} < 80%)")
    print("=" * width)
    print()


if __name__ == "__main__":
    args = _parse_args()

    try:
        metrics = evaluate_sentiment_model(
            model_name=args.model,
            output_dir=args.output_dir,
        )
        _print_report(metrics)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)
