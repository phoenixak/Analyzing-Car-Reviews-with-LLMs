"""The v1 approach, as a baseline on the same test set.

v1 used yangheng/deberta-v3-base-absa-v1.1 (asked about one aspect term at a time) and
DistilBERT-SST2 for the overall sentiment. The ABSA model always returns a polarity, so it
cannot say an aspect is *not* mentioned: every aspect counts as mentioned.
"""

import numpy as np
import torch
from transformers import pipeline

from review_intel.schema import ASPECTS, OVERALL, SENTIMENTS

TERMS = {
    "price": "price",
    "performance": "performance",
    "comfort": "comfort",
    "reliability": "reliability",
    "design": "design",
    "safety": "safety",
    "fuel_economy": "fuel economy",
    "technology": "technology",
}
MAP = {"positive": "positive", "negative": "negative", "neutral": "mixed"}


class V1Baseline:
    def __init__(self, threads: int = 16) -> None:
        torch.set_num_threads(threads)
        self.absa = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1", device=-1)
        self.sst2 = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        asp = np.zeros((len(texts), len(ASPECTS)), dtype=int)
        for k, a in enumerate(ASPECTS):
            outs = self.absa([{"text": t, "text_pair": TERMS[a]} for t in texts], batch_size=16, truncation=True)
            asp[:, k] = [SENTIMENTS.index(MAP[o["label"].lower()]) for o in outs]
        ov = [
            OVERALL.index("positive" if o["label"] == "POSITIVE" else "negative")
            for o in self.sst2(texts, batch_size=32, truncation=True)
        ]
        return asp, np.array(ov)
