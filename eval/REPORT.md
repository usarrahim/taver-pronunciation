# Taver — speechocean762 evaluation

Zero-shot evaluation of the Taver scorer against the **speechocean762** benchmark of L2-English speech with expert human pronunciation ratings. The engine is **not trained or tuned on this data** — these numbers measure how well the goodness-of-pronunciation scorer agrees with human judgement out of the box.

- Split: `test`  ·  utterances scored: **30** (skipped 0)
- Throughput: **308 ms / utterance** (FP32 model, CPU, full pipeline incl. G2P)

## Correlation with human scores

| Level | N | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| Utterance accuracy | 30 | 0.351 | 0.180 |
| Word accuracy | 160 | 0.317 | 0.536 |

## Calibration (engine 0–100 → human 0–10)

Least-squares linear fit so the engine score can be reported on the human 0–10 scale:

- Utterance: `human ≈ 0.0160 × engine + 7.381`  ·  mean abs error **0.57** points (of 10)
- Word: `human ≈ 0.0116 × engine + 8.835`  ·  mean abs error **0.29** points (of 10)

## Reliability (utterance)

Mean human accuracy for each engine-score band — monotonic increase means the score is trustworthy as a ranking.

| Engine band | N | Mean engine | Mean human (0–10) |
|---|---:|---:|---:|
|   0-50  | 3 | 39.3 | 8.00 |
|  50-65  | 1 | 55.8 | 9.00 |
|  65-75  | 1 | 70.8 | 7.00 |
|  75-85  | 4 | 80.2 | 9.00 |
|  85-92  | 8 | 89.0 | 8.62 |
|  92-97  | 8 | 94.7 | 9.12 |
|  97-101 | 5 | 99.6 | 8.80 |
