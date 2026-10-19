# Evaluation results

Test set: 330 reviews about 6 car models that never appear in training (Cadillac Lyriq, Hyundai Elantra, Nissan Altima, Porsche Macan, Rivian R1S, Subaru Crosstrek). Latency is the median per review on CPU with 2 threads.

| Model | Aspect macro-F1 | Mention F1 | Sentiment acc. (mentioned) | All 8 aspects right | Overall verdict | Latency |
|---|---:|---:|---:|---:|---:|---:|
| v1: DeBERTa-v3 ABSA + DistilBERT-SST2 | 0.183 | 0.401 | 88.2% | 0.0% | 74.2% | 2366 ms |
| v2: fine-tuned car-review-absa | 0.921 | 0.991 | 95.6% | 92.4% | 90.6% | 184 ms |

v1 cannot say an aspect is *not* mentioned (it always returns a polarity), which is why its mention F1 and exact match are low; its overall sentiment has no 'mixed' class.

## Per aspect (v2)

| Aspect | Macro-F1 | Mention F1 | Sentiment acc. when mentioned |
|---|---:|---:|---:|
| Price & value | 1.000 | 1.000 | 100.0% |
| Performance | 0.951 | 0.982 | 94.3% |
| Comfort | 0.995 | 0.994 | 98.8% |
| Reliability | 0.907 | 0.994 | 95.6% |
| Design | 0.730 | 0.987 | 93.6% |
| Safety | 0.897 | 1.000 | 93.0% |
| Fuel economy | 0.951 | 1.000 | 97.3% |
| Technology | 0.936 | 0.975 | 93.2% |
