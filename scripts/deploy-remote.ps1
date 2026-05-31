# Upload dist\taver-server to a Windows VPS (does not start the server — run run-demo.ps1 on VPS).
# Usage:
#   .\scripts\pack-server.ps1
#   .\scripts\deploy-remote.ps1 -User Administrator
param(
  [Parameter(Mandatory = $true)][string]$User,
  [string]$HostName = "89.168.59.242",
  [string]$RemoteDir = "C:/taver-demo",
  [string]$LocalPack = "dist\taver-server"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Test-Path $LocalPack)) {
  & (Join-Path $PSScriptRoot "pack-server.ps1")
}

$remote = "${User}@${HostName}"
Write-Host "Uploading to ${remote}:${RemoteDir} ..." -ForegroundColor Cyan
ssh $remote "powershell -NoProfile -Command `"Remove-Item -Recurse -Force '$RemoteDir' -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path '$RemoteDir' | Out-Null`""
scp -r "$LocalPack\*" "${remote}:${RemoteDir}/"

Write-Host ""
Write-Host "Uploaded. On the VPS, run (in PowerShell):" -ForegroundColor Green
Write-Host "  cd C:\taver-demo" -ForegroundColor White
Write-Host "  .\run-demo.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Firewall (once, admin PowerShell on VPS):" -ForegroundColor Yellow
Write-Host "  New-NetFirewallRule -DisplayName Taver -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow"
Write-Host ""
Write-Host "Public URL: http://${HostName}:8080" -ForegroundColor Cyan
