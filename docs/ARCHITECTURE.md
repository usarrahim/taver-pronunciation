# Taver Architecture

## Layers

| Layer | Role | Key files |
|-------|------|-----------|
| A — Audio | WASAPI / WAV → mono 16 kHz | `src/audio/*` |
| B — DSP | MFCC, VAD, silence trim | `src/dsp/*` |
| C — AI | wav2vec2 CTC → frame log-probs | `src/ai/*` |
| D — Align | Forced alignment + GOP | `src/align/*` |
| G2P | Text → IPA → vocab ids | `src/g2p/*` |
| Prosody | F0, rate, fluency | `src/prosody/*` |

## Threading (live mode)

1. **Capture thread** — pushes `AudioChunk` into SPSC ring.
2. **DSP thread** — VAD, utterance assembly, pushes pooled `Utterance*`.
3. **Inference thread** — ONNX + scoring, invokes `on_result` callback.

Synchronous mode (`taver_score_wav` / `taver_score_samples`) runs inference on the caller thread without starting capture.

## Memory

- `MemoryPool` for utterances and logit matrices — no heap on hot path.
- Pre-sized trellis buffers in `GopScorer`.

## Extension points

- Swap `IInferenceEngine` implementation.
- Tune `config/calibration.json` without recompile.
- Add words in `data/lexicon.txt`.
