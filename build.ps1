# Build Taver with the MSVC toolchain bundled in Visual Studio.
# Usage:  .\build.ps1            (configure + build, Release)
#         .\build.ps1 -Clean     (wipe the build directory first)
param([switch]$Clean)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found; is Visual Studio installed?" }
$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "No Visual Studio installation found." }

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$cmake  = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja  = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
foreach ($p in @($vcvars, $cmake, $ninja)) {
  if (-not (Test-Path $p)) { throw "Required tool missing: $p" }
}

$build = Join-Path $root "build"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$cfg = "`"$vcvars`" && `"$cmake`" -G Ninja -S `"$root`" -B `"$build`" -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release"
$bld = "`"$vcvars`" && `"$cmake`" --build `"$build`""

Write-Host "==> Configuring Taver..." -ForegroundColor Cyan
cmd /c $cfg
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)" }

Write-Host "==> Building..." -ForegroundColor Cyan
cmd /c $bld
if ($LASTEXITCODE -ne 0) { throw "Build failed ($LASTEXITCODE)" }

Write-Host "==> Done. Binaries in $build\bin" -ForegroundColor Green
