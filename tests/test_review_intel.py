import json
import random

import numpy as np
import pytest
import torch

from review_intel.data.synth import LABEL_SCHEMA, label_prompt, one, sample_spec, write_prompt
from review_intel.infer import Analyzer, split_sentences
from review_intel.metrics import score
from review_intel.schema import ASPECTS, LABELS, N_ASPECT_LOGITS, OVERALL, SENTIMENTS
from review_intel.train import decode, grouped_loss, split_by_car, targets


def test_label_layout():
    assert len(LABELS) == N_ASPECT_LOGITS + len(OVERALL) == 35
    assert LABELS[:4] == ["price:none", "price:positive", "price:negative", "price:mixed"]


def test_specs_are_consistent():
    rng = random.Random(1)
    for _ in range(200):
        s = sample_spec(rng)
        assert 1 <= len(s["aspects"]) <= 4 and set(s["aspects"]) <= set(ASPECTS)
        pos = sum(v == "positive" for v in s["aspects"].values())
        neg = sum(v == "negative" for v in s["aspects"].values())
        assert s["overall"] == ("positive" if pos > neg else "negative" if neg > pos else "mixed")


def test_prompts_name_wanted_and_banned_aspects():
    spec = sample_spec(random.Random(3))
    user = write_prompt(spec)[1]["content"]
    for a in spec["aspects"]:
        assert a.replace("_", " ") in user
    banned = user.split("Do not say anything about:")[1]
    assert all(a.replace("_", " ") in banned for a in ASPECTS if a not in spec["aspects"])
    assert "none" in label_prompt("Great car.")[0]["content"]
    assert set(LABEL_SCHEMA["properties"]["aspects"]["required"]) == set(ASPECTS)


class FakeTeacher:
    def __init__(self, labels):
        self.labels = labels

    def chat(self, messages, temperature, schema=None, max_tokens=400):
        return json.dumps(self.labels) if schema else "The seats are comfortable and it was cheap."


def test_blind_relabel_keeps_only_agreement():
    spec = {
        "make": "Honda",
        "model": "Civic",
        "body": "car",
        "year": 2020,
        "persona": "p",
        "length": "short",
        "style": "s",
        "aspects": {"comfort": "positive", "price": "positive"},
        "overall": "positive",
    }
    want = {a: spec["aspects"].get(a, "none") for a in ASPECTS}
    assert one(FakeTeacher({"aspects": want, "overall": "positive"}), spec)["agree"] is True
    wrong = dict(want, safety="negative")
    assert one(FakeTeacher({"aspects": wrong, "overall": "positive"}), spec)["agree"] is False
    # the overall verdict comes from the blind reading, not the spec
    row = one(FakeTeacher({"aspects": want, "overall": "mixed"}), spec)
    assert row["agree"] is True and row["overall"] == "mixed" and row["spec_overall"] == "positive"


def test_split_by_car_has_no_shared_cars():
    rows = [{"car": f"Make Model{i % 30}", "text": str(i)} for i in range(300)]
    train, val, test = split_by_car(rows)
    cars = [{r["car"] for r in part} for part in (train, val, test)]
    assert not (cars[0] & cars[1]) and not (cars[0] & cars[2]) and not (cars[1] & cars[2])
    assert len(train) + len(val) + len(test) == 300


def test_loss_and_decode_shapes():
    rows = [{"aspects": {a: "none" for a in ASPECTS} | {"price": "negative"}, "overall": "negative"}] * 4
    y = targets(rows)
    assert y.shape == (4, 9) and int(y[0, 0]) == SENTIMENTS.index("negative")
    logits = torch.zeros(4, len(LABELS), requires_grad=True)
    loss = grouped_loss(logits, y)
    loss.backward()
    asp, ov = decode(logits.detach())
    assert asp.shape == (4, 8) and ov.shape == (4,)


def test_metrics_perfect_and_mention_errors():
    y = np.array([[1, 0, 0, 0, 0, 0, 0, 2], [0, 3, 0, 0, 0, 0, 1, 0]])
    perfect = score(y, y, np.array([2, 1]), np.array([2, 1]))
    assert perfect["exact_match"] == 1.0 and perfect["overall_accuracy"] == 1.0
    p = y.copy()
    p[0, 0] = 0  # missed a mention
    worse = score(y, p)
    assert worse["mention_f1"] < 1.0 and worse["exact_match"] == 0.5


def test_sentence_split():
    assert split_sentences("Great car. Bad tech! Is it worth it? Yes.") == [
        "Great car.",
        "Bad tech!",
        "Is it worth it?",
        "Yes.",
    ]
    assert split_sentences("It does 34 mpg on the highway") == ["It does 34 mpg on the highway"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(
        {
            w
            for w in [
                "the",
                "seats",
                "are",
                "comfortable",
                "and",
                "it",
                "was",
                "cheap",
                "great",
                "car",
                "bad",
                "tech",
                ".",
                "!",
            ]
        }
    )
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    BertTokenizerFast(str(d / "vocab.txt")).save_pretrained(d)
    cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=len(LABELS),
    )
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(d)
    return str(d)


def test_analyzer_end_to_end_without_downloads(tiny_model):
    a = Analyzer(tiny_model, entities=False, min_confidence=0.0)
    r = a.analyze("The seats are comfortable. It was cheap!")
    assert r.overall in OVERALL and len(r.sentences) == 2
    assert all(x.aspect in ASPECTS and x.sentiment in SENTIMENTS[1:] for x in r.aspects)
    assert all(set(x.evidence) <= {0, 1} for x in r.aspects)
    batch = a.analyze_batch(["Great car.", "Bad tech."])
    assert [b.text for b in batch] == ["Great car.", "Bad tech."]
