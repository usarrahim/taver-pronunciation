"""Sanity check: greedy-CTC decode of the exported ONNX phoneme model.

Usage: python tools/decode_check.py models/phoneme.onnx tests/assets/tree.wav ...
Prints the recognized phoneme string per file so we can confirm the model and
the export are correct before wiring it into C++.
"""
import struct
import sys
import wave

import numpy as np
import onnxruntime as ort


def read_wav_mono16k(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    assert sw == 2, "expect 16-bit PCM"
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:  # crude linear resample, fine for a sanity check
        t = np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr))
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


def load_vocab(path):
    id2sym, blank = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if tok.startswith("blank_id="):
                        blank = int(tok.split("=")[1])
                continue
            i, s = line.rstrip("\n").split("\t")
            id2sym[int(i)] = s
    return id2sym, blank


def main():
    model, vocab = sys.argv[1], "models/phoneme_vocab.txt"
    id2sym, blank = load_vocab(vocab)
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    for wav in sys.argv[2:]:
        x = read_wav_mono16k(wav)
        x = (x - x.mean()) / (x.std() + 1e-7)
        logits = sess.run(None, {iname: x[None, :].astype(np.float32)})[0][0]
        ids = logits.argmax(-1)
        # collapse repeats + drop blanks (greedy CTC)
        out, prev = [], -1
        for i in ids:
            if i != prev and i != blank:
                out.append(id2sym.get(int(i), "?"))
            prev = i
        print(f"{wav:32s} -> {' '.join(out)}")


if __name__ == "__main__":
    main()
