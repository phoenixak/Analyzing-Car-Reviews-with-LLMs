"""Analyse a car review: aspect sentiment, the sentences behind each aspect, entities and a verdict.

from review_intel.infer import Analyzer
a = Analyzer("Phoenixak99/car-review-absa")
a.analyze("The CR-V is comfortable but the infotainment keeps freezing.")
"""

import re
from dataclasses import dataclass, field
from functools import cached_property

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_intel.schema import ASPECTS, N_ASPECT_LOGITS, OVERALL, SENTIMENTS

ENTITY_LABELS = ["car make", "car model", "car part", "place"]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


@dataclass
class AspectResult:
    aspect: str
    sentiment: str  # positive / negative / mixed
    confidence: float
    evidence: list[int] = field(default_factory=list)  # indexes into Analysis.sentences


@dataclass
class Analysis:
    text: str
    overall: str
    overall_confidence: float
    aspects: list[AspectResult]
    sentences: list[str]
    entities: list[dict] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


class Analyzer:
    def __init__(
        self,
        model: str = "Phoenixak99/car-review-absa",
        entities: bool = True,
        threads: int | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        if threads:
            torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        self.use_entities = entities
        self.min_confidence = min_confidence

    @cached_property
    def gliner(self):
        from gliner import GLiNER

        return GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    @torch.no_grad()
    def _probs(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
        logits = self.model(**enc).logits.float()
        asp = logits[:, :N_ASPECT_LOGITS].reshape(-1, len(ASPECTS), len(SENTIMENTS)).softmax(-1)
        return asp, logits[:, N_ASPECT_LOGITS:].softmax(-1)

    def analyze(self, text: str) -> Analysis:
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[Analysis]:
        asp, ov = self._probs(texts)
        results = []
        for i, text in enumerate(texts):
            aspects = []
            for k, a in enumerate(ASPECTS):
                p = asp[i, k]
                j = int(p.argmax())
                if j and float(p[j]) >= self.min_confidence:
                    aspects.append(AspectResult(a, SENTIMENTS[j], float(p[j])))
            o = int(ov[i].argmax())
            results.append(Analysis(text, OVERALL[o], float(ov[i, o]), aspects, split_sentences(text)))
        self._attach_evidence(results)
        if self.use_entities:
            for r in results:
                r.entities = [
                    {
                        "text": e["text"],
                        "label": e["label"],
                        "start": e["start"],
                        "end": e["end"],
                        "score": round(float(e["score"]), 3),
                    }
                    for e in self.gliner.predict_entities(r.text, ENTITY_LABELS, threshold=0.45)
                ]
        return results

    def _attach_evidence(self, results: list[Analysis]) -> None:
        """Score each sentence on its own; a sentence is evidence for an aspect it mentions on its own."""
        flat = [(i, s) for i, r in enumerate(results) for s in range(len(r.sentences)) if len(r.sentences) > 1]
        if not flat:
            for r in results:
                for a in r.aspects:
                    a.evidence = [0]
            return
        asp, _ = self._probs([results[i].sentences[s] for i, s in flat])
        for row, (i, s) in enumerate(flat):
            for a in results[i].aspects:
                k = list(ASPECTS).index(a.aspect)
                if float(asp[row, k, 0]) < 0.5:  # this sentence alone mentions the aspect
                    a.evidence.append(s)
        for r in results:
            for a in r.aspects:
                if not a.evidence and len(r.sentences) == 1:
                    a.evidence = [0]
