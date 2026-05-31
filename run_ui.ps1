# Launch the Taver marketing + live demo web UI.
param(
  [string]$Model = "models\phoneme.int8.onnx",
  [string]$Vocab = "models\phoneme_vocab.txt",
  [int]$Port = 8080
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$exe  = Join-Path $root "build\bin\taver_server.exe"
$espeak = Join-Path $root "third_party\espeak\eSpeak NG\espeak-ng.exe"
if (-not (Test-Path $exe)) { throw "Build first: .\build.ps1 (taver_server.exe missing)" }
if (-not (Test-Path (Join-Path $root $Model))) { throw "Model not found: $Model (run .\scripts\setup.ps1)" }
if (-not (Test-Path $espeak)) { Write-Host "Note: espeak-ng not found; free-text targets disabled." -ForegroundColor Yellow }

Start-Process "http://127.0.0.1:$Port"
Write-Host "Taver demo -> http://127.0.0.1:$Port" -ForegroundColor Green
& $exe --host 127.0.0.1 --model (Join-Path $root $Model) --vocab (Join-Path $root $Vocab) --espeak $espeak --root (Join-Path $root "web") --port $Port
