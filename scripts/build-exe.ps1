#requires -Version 5.1
<#
.SYNOPSIS
  Compile scripts\Launcher.cs into "Run VisualLLm.exe" at the repo root, using the
  csc.exe bundled with the .NET Framework on every Windows box (no SDK install).
#>
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $PSScriptRoot "Launcher.cs"
$out  = Join-Path $repo "Run VisualLLm.exe"

$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) {
    $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path $csc)) {
    throw "csc.exe (.NET Framework compiler) not found under $env:WINDIR\Microsoft.NET."
}

Write-Host "Compiling -> $out" -ForegroundColor Cyan
& $csc /nologo /target:exe /optimize+ ("/out:{0}" -f $out) $src
if ($LASTEXITCODE -ne 0) { throw "csc.exe failed ($LASTEXITCODE)." }
Write-Host "Done: $out" -ForegroundColor Green
