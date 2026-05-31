"""Turn eval/report.json into a human-readable REPORT.md with calibration.

Reads the raw (engine, human) score pairs dumped by eval_speechocean.py, then:
  * reports Pearson + Spearman correlation (utterance and word level),
  * fits a linear calibration  human ≈ a * engine + b  (least squares) so the
    0..100 engine score can be mapped onto the human 0..10 scale, and reports
    the mean absolute error of that mapping,
  * prints a reliability table (binned engine score vs mean human score).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    mae = float(np.mean(np.abs((a * x + b) - y)))
    return float(a), float(b), mae


def reliability(engine, human, edges):
    e, h = np.asarray(engine, float), np.asarray(human, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e >= lo) & (e < hi)
        if m.sum() > 0:
            rows.append((f"{lo:>3.0f}-{hi:<3.0f}", int(m.sum()),
                         float(e[m].mean()), float(h[m].mean())))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "eval", "report.json")
    with open(path) as f:
        d = json.load(f)
    rep = d["report"]
    up, uh = d["utt_pred"], d["utt_human"]
    wp, wh = d["word_pred"], d["word_human"]

    ua, ub, umae = linfit(up, uh)
    wa, wb, wmae = linfit(wp, wh)
    rel = reliability(up, uh, [0, 50, 65, 75, 85, 92, 97, 101])

    md = []
    md.append("# Taver — speechocean762 evaluation\n")
    md.append("Zero-shot evaluation of the Taver scorer against the "
              "**speechocean762** benchmark of L2-English speech with expert human "
              "pronunciation ratings. The engine is **not trained or tuned on this "
              "data** — these numbers measure how well the goodness-of-pronunciation "
              "scorer agrees with human judgement out of the box.\n")
    md.append(f"- Split: `{rep['split']}`  ·  utterances scored: **{rep['evaluated']}** "
              f"(skipped {rep['skipped']})")
    md.append(f"- Throughput: **{rep['ms_per_utterance']:.0f} ms / utterance** "
              f"(FP32 model, CPU, full pipeline incl. G2P)\n")

    md.append("## Correlation with human scores\n")
    md.append("| Level | N | Pearson r | Spearman ρ |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Utterance accuracy | {rep['utterance']['n']} | "
              f"{rep['utterance']['pearson']:.3f} | {rep['utterance']['spearman']:.3f} |")
    md.append(f"| Word accuracy | {rep['word']['n']} | "
              f"{rep['word']['pearson']:.3f} | {rep['word']['spearman']:.3f} |\n")

    md.append("## Calibration (engine 0–100 → human 0–10)\n")
    md.append("Least-squares linear fit so the engine score can be reported on the "
              "human 0–10 scale:\n")
    md.append(f"- Utterance: `human ≈ {ua:.4f} × engine + {ub:.3f}`  ·  "
              f"mean abs error **{umae:.2f}** points (of 10)")
    md.append(f"- Word: `human ≈ {wa:.4f} × engine + {wb:.3f}`  ·  "
              f"mean abs error **{wmae:.2f}** points (of 10)\n")

    md.append("## Reliability (utterance)\n")
    md.append("Mean human accuracy for each engine-score band — monotonic increase "
              "means the score is trustworthy as a ranking.\n")
    md.append("| Engine band | N | Mean engine | Mean human (0–10) |")
    md.append("|---|---:|---:|---:|")
    for band, n, me, mh in rel:
        md.append(f"| {band} | {n} | {me:.1f} | {mh:.2f} |")
    md.append("")

    out = os.path.join(ROOT, "eval", "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
