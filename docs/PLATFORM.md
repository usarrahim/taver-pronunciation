# Platform & Deployment

## Shipped today

| Platform | Status |
|----------|--------|
| Windows x64 | ✅ Primary target |
| CPU inference | ✅ Default |
| NVIDIA CUDA | ✅ If ORT GPU package installed |
| DirectML (any D3D12 GPU) | ✅ Dynamic load |

## Planned ports

| Platform | Approach |
|----------|----------|
| macOS / Linux | CMake + PortAudio instead of WASAPI |
| iOS / Android | Static lib + platform audio; CoreML / NNAPI EP |
| Web | WASM build of scorer or thin native bridge |
| Cloud API | Optional wrapper — **not** required; edge-first design |

## Packaging

- Ship `taver.dll` + `onnxruntime.dll` + model + espeak data beside your app.
- INT8 model for constrained devices (`models/phoneme.int8.onnx`).
