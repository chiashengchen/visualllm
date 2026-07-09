#requires -Version 5.1
<#
.SYNOPSIS
  Start the VisualLLm stack (MuseTalk avatar server + pipeline) with a durable log
  file per process, wait until each is healthy, and print the URLs + log paths.

.DESCRIPTION
  Starts the MuseTalk avatar server in its `musetalk` conda env and the pipeline in
  the system python, and propagates the avatar knobs (AVATAR_REF / size / fps / lead /
  tail) from .env to the OS environment -- the avatar server reads OS env ONLY (no
  python-dotenv in its conda env), so without this it would use its built-in defaults
  and mismatch the pipeline/transport.

  Each process writes two kinds of log under logs\ :
    <name>.log              structured loguru (rotated, full tracebacks, uvicorn)
    <name>.out/.err.log     raw stdout/stderr (also catches native mediapipe/onnx spew)

.PARAMETER MusetalkPython
  Path to the python.exe of the 'musetalk' conda env (default E:\miniconda3\envs\musetalk).

.EXAMPLE
  .\scripts\run.ps1
#>
param(
    [string]$MusetalkPython = "E:\miniconda3\envs\musetalk\python.exe",
    [string]$FunasrPython = "E:\miniconda3\envs\funasr-stt\python.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$pids = @()
$envFile = Join-Path $repo ".env"

# Read a single KEY=value from .env (trimmed, no inline-comment handling needed for
# the simple values we propagate). Returns $null if absent.
function Get-EnvVal([string]$key) {
    if (-not (Test-Path $envFile)) { return $null }
    $m = Select-String -Path $envFile -Pattern ("^\s*{0}\s*=\s*(.+?)\s*(?:#.*)?$" -f [regex]::Escape($key)) -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value.Trim() }
    return $null
}

# Propagate avatar knobs from .env to BOTH child processes. The MuseTalk server reads
# these from the OS environment ONLY (no .env loading in its conda env); without this
# it would use its built-in defaults and mismatch the pipeline/transport.
function Set-EnvFromDotenv([string]$key) {
    $v = Get-EnvVal $key
    if ($v) {
        Set-Item -Path ("Env:{0}" -f $key) -Value $v
        Write-Host ("  {0}={1} (from .env -> both processes)" -f $key, $v) -ForegroundColor DarkCyan
    }
}
Write-Host "Avatar engine: musetalk" -ForegroundColor Cyan
Set-EnvFromDotenv "AVATAR_REF"
Set-EnvFromDotenv "MUSETALK_SIZE"
Set-EnvFromDotenv "MUSETALK_BASE_MAX"
Set-EnvFromDotenv "MUSETALK_FPS"
Set-EnvFromDotenv "MUSETALK_LEAD_FRAMES"
Set-EnvFromDotenv "MUSETALK_END_TAIL_FRAMES"
Set-EnvFromDotenv "MUSETALK_IDLE_MOTION"
Set-EnvFromDotenv "MUSETALK_TRT"
Set-EnvFromDotenv "MUSETALK_GPU_COMPOSITE"

# Offline-STT knobs -- the funasr server reads OS env ONLY (like the avatar server).
# sherpa is in-process (system Python), so it needs no env propagation or server here.
$sttProvider = (Get-EnvVal "STT_PROVIDER")
if (-not $sttProvider) { $sttProvider = "deepgram" }
$sttProvider = $sttProvider.ToLower()
if ($sttProvider -eq "funasr") {
    Set-EnvFromDotenv "FUNASR_MODEL"
    Set-EnvFromDotenv "FUNASR_DEVICE"
}

function Test-PortBusy([int]$port) {
    # netstat (native), NOT Get-NetTCPConnection: the CIM cmdlet hangs tens of
    # seconds under CPU load on this box (the windows-process-tools issue; the
    # config panel uses netstat for the same reason).
    $needle = ":{0} " -f $port
    $out = netstat -ano | Select-String -SimpleMatch $needle -ErrorAction SilentlyContinue
    return [bool]($out | Where-Object { $_ -match 'LISTENING' })
}

# Refuse to start over a port that is already taken (the silent bind error that
# wasted time before). 7860 = client, 8002 = avatar server.
foreach ($p in @(7860, 8002)) {
    if (Test-PortBusy $p) {
        Write-Host "ERROR: port $p is already in use. Stop the existing process first." -ForegroundColor Red
        exit 1
    }
}

# 1) MuseTalk avatar server -- own conda env, unbuffered (-u) so logs are live.
if (-not (Test-Path $MusetalkPython)) {
    Write-Host "ERROR: musetalk python not found at $MusetalkPython (pass -MusetalkPython <path>)." -ForegroundColor Red
    exit 1
}
Write-Host "Starting musetalk server -> logs\musetalk.out.log"
$av = Start-Process -FilePath $MusetalkPython `
    -ArgumentList '-u', '-m', 'local_services.musetalk_server.app' `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logs "musetalk.out.log") `
    -RedirectStandardError  (Join-Path $logs "musetalk.err.log")
$pids += $av.Id
Write-Host ("  musetalk PID {0}; waiting for models to load (/health)..." -f $av.Id)
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    try {
        # 127.0.0.1 (not localhost): the server binds IPv4-only, and Windows
        # resolves localhost to ::1 (IPv6) first, which would never answer.
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8002/health" -TimeoutSec 3
        if ($h.ok) { $ok = $true; break }
    } catch { }
}
if ($ok) { Write-Host "  musetalk ready." -ForegroundColor Green }
else { Write-Host "  musetalk not healthy in ~120s; check logs\musetalk.err.log" -ForegroundColor Yellow }

# 1b) Optional local OFFLINE STT (SenseVoice on CPU, ~0 VRAM) -- only when STT_PROVIDER=funasr.
# (sherpa STT is in-process in the pipeline, so it needs no server here.)
if ($sttProvider -eq "funasr") {
    if (-not (Test-Path $FunasrPython)) {
        Write-Host "ERROR: funasr-stt python not found at $FunasrPython (pass -FunasrPython <path>)." -ForegroundColor Red
        exit 1
    }
    Write-Host "Starting funasr STT server -> logs\funasr.out.log"
    $stt = Start-Process -FilePath $FunasrPython `
        -ArgumentList '-u', '-m', 'uvicorn', 'local_services.funasr_server.app:app', '--host', '0.0.0.0', '--port', '8004' `
        -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logs "funasr.out.log") `
        -RedirectStandardError  (Join-Path $logs "funasr.err.log")
    $pids += $stt.Id
    Write-Host ("  funasr PID {0}; waiting for SenseVoice to load (/health)..." -f $stt.Id)
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:8004/health" -TimeoutSec 3
            if ($h.status -eq 'ok') { $ok = $true; break }
        } catch { }
    }
    if ($ok) { Write-Host "  funasr ready." -ForegroundColor Green }
    else { Write-Host "  funasr not healthy in ~120s; check logs\funasr.err.log" -ForegroundColor Yellow }
}

# 2) Pipeline (serves /client at :7860).
Write-Host "Starting pipeline -> logs\pipeline.out.log"
$pipe = Start-Process -FilePath "python" `
    -ArgumentList '-m', 'pipeline.main' `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $logs "pipeline.out.log") `
    -RedirectStandardError  (Join-Path $logs "pipeline.err.log")
$pids += $pipe.Id
Write-Host ("  Pipeline PID {0}; waiting for /client..." -f $pipe.Id)
Start-Sleep -Seconds 2   # let the process import Pipecat before the first probe
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:7860/client/" -TimeoutSec 2 -UseBasicParsing -DisableKeepAlive
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if ($ok) { Write-Host "  Pipeline ready." -ForegroundColor Green }
else { Write-Host "  Client not up yet; check logs\pipeline.err.log" -ForegroundColor Yellow }

Write-Host ""
Write-Host "== VisualLLm running ==" -ForegroundColor Cyan
Write-Host "  Client : http://localhost:7860/client/"
Write-Host "  Logs (structured): $logs\pipeline.log  $logs\musetalk.log"
Write-Host "  Raw stdout/stderr (native spew + banner): logs\*.out.log / *.err.log"
Write-Host ""
Write-Host ("Stop with:  Stop-Process -Id {0}" -f ($pids -join ','))
