# Car Review Intelligence — v2

[![CI](https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-F4F1EA?labelColor=141310)](https://huggingface.co/spaces/Phoenixak99/car-reviews-analyzer)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-car--review--absa-F4F1EA?labelColor=141310)](https://huggingface.co/Phoenixak99/car-review-absa)

What did the owner actually say about the car? One small model reads a review and returns the reviewer's opinion on **eight aspects** — price, performance, comfort, reliability, design, safety, fuel economy, technology — plus an overall verdict, the sentences behind each label, and the cars and parts mentioned. It runs in **184 ms on two CPU threads**.

**[Try the live demo →](https://huggingface.co/spaces/Phoenixak99/car-reviews-analyzer)** — paste a review, or upload a CSV for a heat-map of praise and complaints across hundreds of reviews.

## Results

330 test reviews about six car models that never appear in training (Cadillac Lyriq, Hyundai Elantra, Nissan Altima, Porsche Macan, Rivian R1S, Subaru Crosstrek). Reproduce with `python -m review_intel.evaluate`.

| Model | Aspect macro-F1 | Mention F1 | Sentiment acc. (mentioned) | All 8 aspects right | Overall verdict | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| v1: DeBERTa-v3 ABSA + DistilBERT-SST2 | 0.183 | 0.401 | 88.2% | 0.0% | 74.2% | 2,366 ms |
| **v2: fine-tuned ModernBERT** | **0.921** | **0.991** | **95.6%** | **92.4%** | **90.6%** | **184 ms** |

v1 asked a general ABSA model about each aspect in turn. It always returns a polarity, so it can't say an aspect isn't discussed, and it needs eight forward passes per review. v2 answers everything in one pass, 13× faster. Per-aspect results are in [`eval/results.md`](eval/results.md); design is the weakest (macro-F1 0.73) because *mixed* design opinions are rare.

## How it was built

There was no car-review dataset with a clean licence: the public sets are scraped from review sites, including one that relabels Edmunds content as Apache-2.0. So the training data is **synthetic, labelled by construction, and checked twice**:

```
 sample a spec ──▶ teacher writes the review ──▶ teacher re-labels it blind ──▶ keep if the labels match
 car · persona ·     Qwen3-8B, llama.cpp on         JSON-schema-constrained       3,872 of 6,000 kept
 tone · length ·     an RX 9070 XT (Vulkan),        output, spec hidden           (64.5%)
 1–4 aspects ±       ~52 reviews / minute
                                      │
                      ModernBERT-base, 35-way head ◀── split by car model (train / val / test)
                      one cross-entropy per aspect + overall verdict
```

- **Spot check:** 40 random kept reviews read by hand: 35 exactly right; the other five each missed a faint passing mention, none flipped a sentiment.
- **Overall verdict** comes from the blind reading of the finished review (a tie such as "comfortable but ugly" can still read as positive overall).
- **Evidence:** the model also scores each sentence on its own; a sentence that mentions an aspect by itself is shown as the evidence for that label.
- **Entities:** GLiNER-small (zero-shot) marks car makes, models, parts and places.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[demo,data,dev]"

python space/app.py                                    # demo on http://127.0.0.1:7860 (pulls the model from the Hub)
pytest -q                                              # 9 tests, tiny in-memory model, no downloads

# rebuild everything from scratch
llama-server -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 32768 -np 8 --jinja --port 8088
python -m review_intel.data.synth --n 6000             # → data/synthetic.jsonl (included in the repo)
python -m review_intel.train --epochs 3                # → models/car-review-absa
python -m review_intel.evaluate                        # → eval/results.md
```

```python
from review_intel.infer import Analyzer

a = Analyzer("Phoenixak99/car-review-absa")
r = a.analyze("Roomy and comfortable, but the touchscreen freezes every week.")
r.overall                                         # 'mixed'
[(x.aspect, x.sentiment) for x in r.aspects]      # [('comfort', 'positive'), ('technology', 'negative')]
```

## Limitations

Trained and tested on synthetic English reviews: real reviews with heavy slang, sarcasm or long off-topic stretches are harder, and accuracy on them is unmeasured. Car names are real but the opinions in the data are invented. Faint, implicit mentions are the most common miss.

## What changed from v1

v1 (tagged [`v1`](https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs/tree/v1)) chained seven off-the-shelf pipelines (SST-2 sentiment, BERTopic, T5 summaries, opus-mt translation, extractive QA, CoNLL NER, DeBERTa ABSA) with no shared evaluation. v2 does one job end to end — aspect-level sentiment — with its own data pipeline, a fine-tuned model published on the Hub, a measured 5× macro-F1 gain over v1's approach at 1/13th of the latency, tests and CI.

## Licence

Code MIT; model Apache-2.0 (from ModernBERT-base, Apache-2.0, and data generated with Qwen3-8B, Apache-2.0). `data/synthetic.jsonl` is synthetic text released with the code.
