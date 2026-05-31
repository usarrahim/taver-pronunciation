# Contributing to Taver

Thank you for your interest in Taver.

## Development setup

1. Windows 10/11 with Visual Studio 2022+ (Desktop development with C++).
2. `.\scripts\setup.ps1` — ONNX Runtime, espeak-ng, Python venv, model export.
3. `.\build.ps1` — builds `taver.dll`, `taver_cli.exe`, `taver_server.exe`, tests.

## Code style

- C++20, minimal diffs, match existing naming (`taver::` namespace, `taver_*` C API).
- No heap allocations on the audio hot path.
- Comments only for non-obvious logic.

## Pull requests

1. Describe the problem and solution.
2. Run `.\build.ps1` and unit tests under `build\bin\`.
3. Do not commit large binaries (`models/*.onnx`, `third_party/`).

## Reporting issues

Include OS version, build command output, and steps to reproduce.
