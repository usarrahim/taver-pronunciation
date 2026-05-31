# Quick latency benchmark for Taver (Phase 3 tooling).
param(
  [string]$Wav = "samples\three.wav",
  [string]$Text = "three",
  [int]$Loops = 20
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path -Parent
$cli = Join-Path $root "build\bin\taver_cli.exe"
if (-not (Test-Path $cli)) { throw "Build first: .\build.ps1" }
$times = @()
for ($i = 0; $i -lt $Loops; $i++) {
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  & $cli --text $Text --wav (Join-Path $root $Wav) 2>$null | Out-Null
  $sw.Stop()
  $times += $sw.ElapsedMilliseconds
}
$avg = ($times | Measure-Object -Average).Average
$p50 = ($times | Sort-Object)[[int]($Loops * 0.5)]
Write-Host "Taver benchmark: $Loops runs, avg=$([math]::Round($avg))ms p50=$p50 ms"
