"""Fine-tune an encoder into an aspect-sentiment model (8 aspects × 4 states + overall verdict).

    python -m review_intel.train --data data/synthetic.jsonl --out models/car-review-absa

The model is a plain AutoModelForSequenceClassification with 35 outputs, trained with one
cross-entropy per aspect group plus one for the overall verdict, so it loads anywhere.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from review_intel.metrics import score
from review_intel.schema import ASPECTS, LABELS, N_ASPECT_LOGITS, OVERALL, SENTIMENTS


def load_rows(path: str) -> list[dict]:
    rows, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = r["text"].strip().lower()
        if r.get("agree") and key not in seen:  # keep only labels the teacher agreed with twice
            seen.add(key)
            rows.append(r)
    return rows


def split_by_car(rows: list[dict], seed: int = 13) -> tuple[list, list, list]:
    """80/10/10 by car model, so test reviews are about cars the model never saw in training."""
    cars = sorted({r["car"] for r in rows})
    random.Random(seed).shuffle(cars)
    n = len(cars)
    test_cars, val_cars = set(cars[: max(1, n // 10)]), set(cars[n // 10 : max(2, n // 5)])
    test = [r for r in rows if r["car"] in test_cars]
    val = [r for r in rows if r["car"] in val_cars]
    train = [r for r in rows if r["car"] not in test_cars | val_cars]
    return train, val, test


def targets(rows: list[dict]) -> torch.Tensor:
    """(n, 9) class indices: one per aspect, then the overall verdict."""
    return torch.tensor(
        [[SENTIMENTS.index(r["aspects"][a]) for a in ASPECTS] + [OVERALL.index(r["overall"])] for r in rows]
    )


def grouped_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    asp = logits[:, :N_ASPECT_LOGITS].reshape(-1, len(ASPECTS), len(SENTIMENTS))
    loss = F.cross_entropy(asp.reshape(-1, len(SENTIMENTS)), y[:, : len(ASPECTS)].reshape(-1))
    return loss + 0.5 * F.cross_entropy(logits[:, N_ASPECT_LOGITS:], y[:, -1])


def decode(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    asp = logits[:, :N_ASPECT_LOGITS].reshape(-1, len(ASPECTS), len(SENTIMENTS)).argmax(-1).numpy()
    return asp, logits[:, N_ASPECT_LOGITS:].argmax(-1).numpy()


@torch.no_grad()
def predict(model, tok, texts: list[str], batch: int = 32, max_len: int = 256) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        out.append(model(**enc).logits.float())
    return torch.cat(out)


def evaluate(model, tok, rows: list[dict]) -> dict:
    asp, ov = decode(predict(model, tok, [r["text"] for r in rows]))
    y = targets(rows).numpy()
    return score(y[:, : len(ASPECTS)], asp, y[:, -1], ov)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic.jsonl")
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out", default="models/car-review-absa")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rows = load_rows(args.data)
    train, val, test = split_by_car(rows, args.seed)
    print(f"{len(rows)} agreed reviews → train {len(train)} · val {len(val)} · test {len(test)} (split by car model)")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) / args.batch) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    y_train = targets(train)

    best, out = -1.0, Path(args.out)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        t0, total = time.time(), 0.0
        for step, i in enumerate(range(0, len(order), args.batch), start=1):
            idx = order[i : i + args.batch]
            enc = tok(
                [train[j]["text"] for j in idx],
                truncation=True,
                max_length=args.max_len,
                padding=True,
                return_tensors="pt",
            )
            loss = grouped_loss(model(**enc).logits, y_train[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step} loss {total / step:.4f} ({(time.time() - t0) / step:.2f}s/step)",
                    flush=True,
                )
        m = evaluate(model, tok, val)
        print(
            f"epoch {epoch} val: aspect macro-F1 {m['aspect_macro_f1']:.3f} · overall acc {m['overall_accuracy']:.3f}",
            flush=True,
        )
        if m["aspect_macro_f1"] > best:
            best = m["aspect_macro_f1"]
            model.save_pretrained(out)
            tok.save_pretrained(out)

    model = AutoModelForSequenceClassification.from_pretrained(out)
    test_m = evaluate(model, tok, test)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "val_best_aspect_macro_f1": best,
                "test": test_m,
                "sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "base": args.base,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Path("data/test_cars.json").write_text(json.dumps(sorted({r["car"] for r in test})), encoding="utf-8")
    print(json.dumps(test_m, indent=2))


if __name__ == "__main__":
    main()
