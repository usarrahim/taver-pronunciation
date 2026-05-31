# Changelog

## [1.0.0] — 2026-05-31

### Added
- **Taver** product branding (renamed from internal Promptu codename).
- Phoneme-level GOP scoring engine (`taver.dll`) with C API.
- Arbitrary English text targets via espeak-ng G2P.
- Word-level and prosody metrics.
- Marketing + live demo web UI (`web/index.html`).
- speechocean762 evaluation harness.
- Configurable GOP calibration (`config/calibration.json`).
- Pronunciation lexicon overrides (`data/lexicon.txt`).
- US / UK accent selection for G2P.
- Browser session history in the demo UI.
- DirectML + CUDA + CPU ONNX Runtime backends.
- Light-theme marketing site, coach notes, `web/logo.svg`, footprint section.
- INT8 default for demo; `docs/DEPLOYMENT.md` (~350 MB app bundle).

### Fixed
- G2P subprocess leak on long eval runs (terminate hung espeak).
- MSVC `NOMINMAX` conflict with `std::max` in ONNX inference.
