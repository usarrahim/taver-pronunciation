# Taver

**Your on-device AI pronunciation coach** — scores every **sound**, not just the word.

Taver is built for **language-learning apps** and classroom tools: embed `taver.dll`, ship the **INT8 model (~300 MB)**, and give learners teacher-style feedback (which phoneme was wrong, swallowed, or weak) — fully offline.

```
Audio → wav2vec2 (ONNX) → CTC alignment → GOP scores → coach feedback
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-live-3b6fd9)](http://89.168.59.242:8080)

**Try it:** [http://89.168.59.242:8080](http://89.168.59.242:8080) — live pronunciation coach (microphone required, scored on the server).

---

## Download size (important)

| Who | What | Size |
|-----|------|------|
| **Learner / your app** | INT8 model + engine + espeak | **~350 MB** (one-time in installer) |
| **Developer** | `setup.ps1` exports FP32+INT8 from Hugging Face | ~1.2 GB **once** on dev machine |

Learners do **not** download 1–2 GB per session. The large file is only for **building** the model. Production apps ship **`phoneme.int8.onnx`**.

Details: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## Why this is not “just another STT score”

| Speech-to-text apps | Taver (phoneme coach) |
|---------------------|------------------------|
| “Did you say the right **word**?” | “Did you produce the right **sounds**?” |
| *tree* / *three* often both “OK” | /t/ vs /θ/ flagged explicitly |
| Cloud transcription | **100% on-device** |
| Opaque % score | Per-phoneme verdict + coach tips |

**Value for L2 English:** minimal pairs, consonant clarity, connected speech — what a human tutor checks first.

---

## Quick start

```powershell
# 1) One-time dev setup (downloads ~1.2 GB to export models)
.\scripts\setup.ps1

# 2) Build
.\build.ps1

# 3) Live demo (local)
.\run_ui.ps1
# → http://127.0.0.1:8080

# Public demo (same UI, hosted):
# http://89.168.59.242:8080

# 4) CLI (INT8 recommended for production-like footprint)
.\build\bin\taver_cli.exe --model models\phoneme.int8.onnx --text "three" --wav samples\three.wav
```

---

## Integrate in your app

```c
#include "taver/taver.h"

TaverConfig cfg = {
  .model_path = "models/phoneme.int8.onnx",   // ~300 MB — ship in installer
  .vocab_path = "models/phoneme_vocab.txt",
  .espeak_path = "espeak-ng.exe",
  .lexicon_path = "data/lexicon.txt",
  .calibration_path = "config/calibration.json",
  .backend = TAVER_BACKEND_AUTO,
};
TaverEngine* eng;
taver_create(&cfg, &eng);
taver_set_target_text(eng, "I would like some water");
TaverResult r;
taver_score_samples(eng, pcm_float32, num_samples, &r);
// r.phonemes[] — per-sound scores + error types for your UI
taver_destroy(eng);
```

See [docs/LMS.md](docs/LMS.md) for LMS / xAPI notes.

---

## Deploy public demo (Windows VPS)

Pack only what the server needs (~350 MB), upload to **89.168.59.242**, listen on all interfaces:

```powershell
# On your dev machine (after build + setup)
.\scripts\pack-server.ps1

# Upload + run (SSH/SCP — OpenSSH client on Windows)
.\scripts\deploy-remote.ps1 -User Administrator -HostName 89.168.59.242

# Or manual upload:
scp -r dist\taver-server\* Administrator@89.168.59.242:C:\taver-demo\
ssh Administrator@89.168.59.242 "powershell -File C:\taver-demo\run-demo.ps1"
```

On the VPS, allow inbound **TCP 8080** in Windows Firewall. Demo URL for README/GitHub:
**http://89.168.59.242:8080**

---

## Features

- Phoneme-level GOP scoring (good / fair / wrong / swallowed)
- Any English text → target phonemes (G2P)
- Word & sentence roll-up + prosody (pitch, rate, fluency)
- US / UK accent selection
- C API · CLI · local demo web UI
- Benchmark harness (speechocean762)

---

## Project layout

```
include/taver/     Public C API
src/               Engine (audio, DSP, AI, align, g2p, prosody)
app/               taver_cli, taver_server
web/               Marketing site + live demo (logo.svg)
config/            calibration.json
data/              lexicon.txt
docs/              DEPLOYMENT, ARCHITECTURE, LMS, ENTERPRISE
```

---

## Tests & validation

```powershell
.\build\bin\test_g2p.exe
.\tools\.venv\Scripts\python.exe tools\eval_speechocean.py --limit 200
.\tools\.venv\Scripts\python.exe tools\make_report.py
```

---

## License

MIT — [LICENSE](LICENSE)
