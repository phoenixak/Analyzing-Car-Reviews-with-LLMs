"""Compare the fine-tuned model with the v1 pipeline on the held-out cars, and time both on CPU.

    python -m review_intel.evaluate --model models/car-review-absa
Writes eval/results.md.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_intel.baseline import V1Baseline
from review_intel.metrics import score
from review_intel.schema import ASPECTS, NICE
from review_intel.train import decode, load_rows, predict, split_by_car, targets


def latency(fn, texts: list[str], n: int = 60) -> float:
    fn(texts[:2])  # warm-up
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        fn([t])
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def row(name: str, m: dict, ms: float) -> str:
    return (
        f"| {name} | {m['aspect_macro_f1']:.3f} | {m['mention_f1']:.3f} | {m['sentiment_acc_when_mentioned']:.1%} | "
        f"{m['exact_match']:.1%} | {m['overall_accuracy']:.1%} | {ms:.0f} ms |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic.jsonl")
    ap.add_argument("--model", default="models/car-review-absa")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    rows = load_rows(args.data)
    _, _, test = split_by_car(rows)
    texts = [r["text"] for r in test]
    y = targets(test).numpy()
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    asp, ov = decode(predict(model, tok, texts))
    ours = score(y[:, : len(ASPECTS)], asp, y[:, -1], ov)
    ours_ms = latency(lambda b: predict(model, tok, b), texts)

    v1 = V1Baseline(threads=args.threads)
    b_asp, b_ov = v1.predict(texts)
    base = score(y[:, : len(ASPECTS)], b_asp, y[:, -1], b_ov)
    base_ms = latency(v1.predict, texts, n=20)

    cars = sorted({r["car"] for r in test})
    lines = [
        "# Evaluation results",
        "",
        f"Test set: {len(test)} reviews about {len(cars)} car models that never appear in training "
        f"({', '.join(cars)}). Latency is the median per review on CPU with {args.threads} threads.",
        "",
        "| Model | Aspect macro-F1 | Mention F1 | Sentiment acc. (mentioned) | All 8 aspects right | Overall verdict | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row("v1: DeBERTa-v3 ABSA + DistilBERT-SST2", base, base_ms),
        row(f"v2: fine-tuned {Path(args.model).name}", ours, ours_ms),
        "",
        "v1 cannot say an aspect is *not* mentioned (it always returns a polarity), which is why its mention F1 and "
        "exact match are low; its overall sentiment has no 'mixed' class.",
        "",
        "## Per aspect (v2)",
        "",
        "| Aspect | Macro-F1 | Mention F1 | Sentiment acc. when mentioned |",
        "|---|---:|---:|---:|",
        *[
            f"| {NICE[a]} | {v['macro_f1']:.3f} | {v['mention_f1']:.3f} | {v['sentiment_acc_when_mentioned']:.1%} |"
            if v["sentiment_acc_when_mentioned"] is not None
            else f"| {NICE[a]} | {v['macro_f1']:.3f} | {v['mention_f1']:.3f} | – |"
            for a, v in ours["per_aspect"].items()
        ],
    ]
    out = Path("eval")
    out.mkdir(exist_ok=True)
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {
                "v2": ours,
                "v1": base,
                "latency_ms": {"v2": ours_ms, "v1": base_ms},
                "test_cars": cars,
                "n_test": len(test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines[:9]))
    print(f"exact-match agreement check: {np.mean((asp == y[:, :8]).all(1)):.3f}")


if __name__ == "__main__":
    main()
