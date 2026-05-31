"""Evaluate Taver against the speechocean762 benchmark.

Drives the real engine (Taver.dll, identical code path to the product) over L2
English recordings with expert human pronunciation scores, and reports how well
the engine's scores correlate with the human judgements at the utterance and word
level (Pearson + Spearman). This is the credibility check for the scorer.

Usage:
  python tools/eval_speechocean.py [--limit N] [--out eval/report.json]
"""
import argparse
import ctypes as C
import io
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "bin")


# ---- ctypes mirror of the public C API ------------------------------------
class PhonemeScore(C.Structure):
    _fields_ = [("phoneme", C.c_char * 16), ("realized", C.c_char * 16),
                ("score", C.c_float), ("error_type", C.c_int),
                ("start_sec", C.c_float), ("duration_sec", C.c_float)]


class WordScore(C.Structure):
    _fields_ = [("text", C.c_char * 40), ("score", C.c_float),
                ("phoneme_start", C.c_int), ("phoneme_count", C.c_int),
                ("has_error", C.c_int)]


class Result(C.Structure):
    _fields_ = [("overall_score", C.c_float), ("num_phonemes", C.c_int),
                ("phonemes", PhonemeScore * 128), ("num_words", C.c_int),
                ("words", WordScore * 48), ("utterance_sec", C.c_float),
                ("latency_ms", C.c_float), ("speaking_rate_pps", C.c_float),
                ("mean_pitch_hz", C.c_float), ("pitch_range_hz", C.c_float),
                ("fluency_score", C.c_float)]


class Config(C.Structure):
    _fields_ = [
        ("model_path", C.c_char_p), ("vocab_path", C.c_char_p),
        ("espeak_path", C.c_char_p), ("lexicon_path", C.c_char_p),
        ("calibration_path", C.c_char_p), ("backend", C.c_int),
        ("g2p_voice", C.c_int), ("print_features", C.c_int),
        ("on_result", C.c_void_p), ("user_data", C.c_void_p),
    ]


def load_lib():
    os.add_dll_directory(BIN)
    lib = C.CDLL(os.path.join(BIN, "taver.dll"))
    lib.taver_create.argtypes = [C.POINTER(Config), C.POINTER(C.c_void_p)]
    lib.taver_create.restype = C.c_int
    lib.taver_set_target_text.argtypes = [C.c_void_p, C.c_char_p]
    lib.taver_set_target_text.restype = C.c_int
    lib.taver_score_samples.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int,
                                        C.POINTER(Result)]
    lib.taver_score_samples.restype = C.c_int
    lib.taver_destroy.argtypes = [C.c_void_p]
    return lib


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        return r
    return pearson(rank(np.asarray(x, float)), rank(np.asarray(y, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "report.json"))
    args = ap.parse_args()

    lib = load_lib()
    cfg = Config()
    cfg.model_path = os.path.join(ROOT, "models", "phoneme.onnx").encode()
    cfg.vocab_path = os.path.join(ROOT, "models", "phoneme_vocab.txt").encode()
    cfg.espeak_path = os.path.join(ROOT, "third_party", "espeak", "eSpeak NG",
                                   "espeak-ng.exe").encode()
    cfg.lexicon_path = os.path.join(ROOT, "data", "lexicon.txt").encode()
    cfg.calibration_path = os.path.join(ROOT, "config", "calibration.json").encode()
    cfg.backend = 0
    cfg.g2p_voice = 0
    eng = C.c_void_p()
    st = lib.taver_create(C.byref(cfg), C.byref(eng))
    if st != 0:
        print("engine create failed:", st); sys.exit(1)

    ds = load_dataset("mispeech/speechocean762", split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    n = min(args.limit, len(ds)) if args.limit > 0 else len(ds)
    print(f"Evaluating {n} / {len(ds)} {args.split} utterances ...")

    utt_pred, utt_human = [], []
    word_pred, word_human = [], []
    skipped = 0
    t0 = time.time()
    res = Result()

    for i in range(n):
        ex = ds[i]
        text = ex["text"]
        if lib.taver_set_target_text(eng, text.encode("utf-8")) != 0:
            skipped += 1; continue
        wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        ptr = wav.ctypes.data_as(C.POINTER(C.c_float))
        if os.environ.get("TAVER_DEBUG"):
            print(f"[{i}] len={len(wav)} words={len(ex['words'])} text={ex['text'][:40]!r}",
                  flush=True)
        if lib.taver_score_samples(eng, ptr, len(wav), C.byref(res)) != 0:
            skipped += 1; continue

        utt_pred.append(res.overall_score)
        utt_human.append(float(ex["accuracy"]))

        hwords = ex["words"]
        if res.num_words == len(hwords):
            for w in range(res.num_words):
                word_pred.append(res.words[w].score)
                word_human.append(float(hwords[w]["accuracy"]))

        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n}  ({dt/(i+1)*1000:.0f} ms/utt)  "
                  f"PCC_utt={pearson(utt_pred, utt_human):.3f}", flush=True)

    lib.taver_destroy(eng)
    dt = time.time() - t0

    report = {
        "split": args.split,
        "evaluated": len(utt_pred),
        "skipped": skipped,
        "ms_per_utterance": dt / max(1, len(utt_pred)) * 1000.0,
        "utterance": {
            "n": len(utt_pred),
            "pearson": pearson(utt_pred, utt_human),
            "spearman": spearman(utt_pred, utt_human),
            "human_range": [min(utt_human), max(utt_human)] if utt_human else None,
        },
        "word": {
            "n": len(word_pred),
            "pearson": pearson(word_pred, word_human),
            "spearman": spearman(word_pred, word_human),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"report": report,
                   "utt_pred": utt_pred, "utt_human": utt_human,
                   "word_pred": word_pred, "word_human": word_human}, f, indent=2)

    print("\n==== speechocean762 evaluation ====")
    print(json.dumps(report, indent=2))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
