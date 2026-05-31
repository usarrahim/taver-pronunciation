# Enterprise & Privacy

## Privacy by design

- **No audio upload** in the default deployment — scoring runs locally.
- Suitable for schools, healthcare-adjacent training, and regulated environments where data must not leave the device.

## Deployment models

1. **Desktop / kiosk** — `taver.dll` + local HTTP UI (`taver_server`).
2. **Embedded in your app** — link C API; your UI owns microphone consent.
3. **Optional server** — only if you add one; not part of the reference product.

## Compliance notes (non-legal)

- Document that espeak-ng and ONNX Runtime are third-party components with their own licenses.
- Provide a DPIA-friendly data flow: mic → RAM → scores → UI (no persistence by default).
- Session history in the demo UI is **browser localStorage only** — disable in production builds if undesired.

## Support expectations

- Calibrate scores on your population using `tools/eval_speechocean.py` or in-house labelled data.
- Adjust `config/calibration.json` for your scoring scale.
