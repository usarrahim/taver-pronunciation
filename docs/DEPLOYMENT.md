# Deployment footprint (for app integrators)

## What end users download

Taver is **not** a 1–2 GB runtime for learners. That size is a **one-time developer setup** to export the model from Hugging Face.

| Component | Typical size | Who ships it |
|-----------|-------------|--------------|
| `taver.dll` + ONNX Runtime | ~15 MB | Your app installer |
| **INT8 model** (`phoneme.int8.onnx`) | **~300 MB** | Your app (recommended) |
| FP32 model (`phoneme.onnx`) | ~1.2 GB | Optional (dev / max accuracy) |
| espeak-ng data + exe | ~25 MB | Your app |
| Vocab + config | <1 MB | Your app |

**Recommended learner bundle: ~350 MB total** — comparable to one offline language pack, not a multi‑GB cloud dependency.

## Integration patterns

1. **Ship INT8 in the installer** — zero download at lesson time.
2. **First-run download** — fetch `phoneme.int8.onnx` once, cache on disk.
3. **LMS / mobile** — same `taver.dll` (or static lib when ported); your UI owns the microphone.

## What runs on device

- Audio stays local (WASAPI / your capture → `taver_score_samples`).
- No STT — phoneme alignment only.
- Scores returned in <200 ms per utterance on modern CPUs.

## Developer vs production

| Step | Size | Frequency |
|------|------|-----------|
| `scripts/setup.ps1` (export FP32+INT8) | ~1.2 GB download | Once per dev machine |
| Your CI packages INT8 only | ~350 MB artifact | Per release |

Use `.\run_ui.ps1` (defaults to INT8) or `taver_cli --model models\phoneme.int8.onnx`.
