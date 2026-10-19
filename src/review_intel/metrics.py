"""Metrics for aspect-sentiment predictions (shared by training, the baseline and the eval script)."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from review_intel.schema import ASPECTS


def score(y_asp: np.ndarray, p_asp: np.ndarray, y_overall=None, p_overall=None) -> dict:
    """y_asp / p_asp: (n, 8) class indices (0 = not mentioned, 1 positive, 2 negative, 3 mixed)."""
    per_aspect = {}
    for k, a in enumerate(ASPECTS):
        yt, pt = y_asp[:, k], p_asp[:, k]
        mentioned = yt > 0
        per_aspect[a] = {
            "macro_f1": float(f1_score(yt, pt, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
            "mention_f1": float(f1_score(yt > 0, pt > 0, zero_division=0)),
            "sentiment_acc_when_mentioned": float(accuracy_score(yt[mentioned], pt[mentioned]))
            if mentioned.any()
            else None,
        }
    yt, pt = y_asp.ravel(), p_asp.ravel()
    m = yt > 0
    out = {
        "aspect_macro_f1": float(np.mean([v["macro_f1"] for v in per_aspect.values()])),
        "mention_f1": float(f1_score(yt > 0, pt > 0, zero_division=0)),
        "sentiment_acc_when_mentioned": float(accuracy_score(yt[m], pt[m])),
        "exact_match": float(np.mean((y_asp == p_asp).all(axis=1))),
        "per_aspect": per_aspect,
    }
    if y_overall is not None:
        out["overall_accuracy"] = float(accuracy_score(y_overall, p_overall))
    return out
