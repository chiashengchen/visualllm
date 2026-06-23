# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **`STATUS.md` is the source of truth** for current state and what's in
> progress — read it first. **`WORKFLOW.md`** is the detailed end-to-end workflow
> (turn flow, avatar wire contract, running locally + remote, full `.env` reference).
> (The parent `E:\Claude\CLAUDE.md` describes a *different* repo, the `.claude` config
> workspace, and does not apply here.)

## What this is

A real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
Multi-turn, streaming end-to-end. Goal: time-to-first-output **< 8 s**. Built on
**Pipecat 1.3.0**, WebRTC to a browser at `/client`.

**Current stack (fully local TTS + avatar). See `STATUS.md` for the full state + the
A/V-sync architecture decision (read it before touching sync).**

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en-US`/`zh-TW`/`th` by `LANGUAGE`) | cloud |
| LLM | OpenRouter (any model via `OPENROUTER_MODEL`) | cloud |
| TTS | **CosyVoice2-0.5B** local streaming server (female zero-shot voice), **on vLLM in WSL** (TTFB ~1.1s; the Windows `tts`-env PyTorch server is the fallback) | **`:8001`, separate repo `E:\Claude\cosyvoice-local-tts`** |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait) | **`:8002`, `musetalk` conda env** |

**TTS note:** CosyVoice runs its autoregressive LLM on **vLLM inside WSL Ubuntu**
(`cosyvllm` conda env on the Blackwell 5060 Ti) — this cut first-chunk latency ~3.4s→~1.1s, the root
cause of the avatar lip-lag. The pipeline reaches it via `COSYVOICE_URL` set to the **WSL IP**, NOT
`localhost` (WSL2's localhost relay buffers the streaming audio ~2s). Run it with
`bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh` in WSL. The original Windows `tts`-env
PyTorch server is the fallback (set `COSYVOICE_URL=http://localhost:8001` + start it). Full build
notes + gotchas: the `project-visualllm-cosyvoice-vllm` memory.

**Shared-GPU VRAM (why "won't talk" can mean CosyVoice crashed):** vLLM and MuseTalk share the one
16GB card. vLLM's `gpu_memory_utilization` (env `COSYVOICE_VLLM_GPU_UTIL`, **default `0.3`**, set in
the cosyvoice repo) must exceed vLLM's own ~4GB footprint or load crashes with "No available memory
for the cache blocks" (the old hardcoded `0.2` = 3.26GB was too low). If the avatar shows but the bot
is silent, first check `:8001` is actually up — the pipeline log shows "Cannot connect to host …:8001".
Free VRAM (close a heavy GPU app) or nudge the util fraction; the "Available KV cache memory" log line
must be positive.

Each stage is a thin single-provider factory in `pipeline/stages/` chosen by `.env` — these
are **deliberate fallback switches, not multi-provider branching**:
- `TTS_PROVIDER` = `cosyvoice` (default) | `elevenlabs` | `deepgram` (the last two are cloud fallbacks).

Core `.env` knobs: `LANGUAGE` (en/zh/th), `TTFO_TARGET_SECONDS`, `TTS_PROVIDER`,
`MUSETALK_SYNC_MODE` (**`steady`** = video-master, synced start, the user's pick and current
default; `live` = audio-master, voice instant + lips trail ~0.75s, can never pause. The old
**steady "screech" is FIXED** — it was pipecat discarding the partial audio buffer after a >3s
render-stall gap (`BOT_VAD_STOP_FALLBACK_SECS`); see `docs/PROBLEMS-AND-FIXES.md` P3 +
`main.py::_relax_bot_vad_stop_timeout` and `musetalk_video.py::_align_even`. Remaining steady
tradeoff: under a long render stall the voice briefly **pauses** then resumes clean — switch to
`live` if that pause is worse than the lip trail), `MUSETALK_FPS` (**12** now; **keep it a divisor
of 16000** — 8/10/16/20/25 — so frame count = audio length. The server's `samples_for_frames` ceil
sizing makes 12 correct anyway; the old `int(16000/fps)` truncation lost ~1 frame/segment → lips
finished ~1–2s early, `docs/PROBLEMS-AND-FIXES.md` P9. NOTE: a leftover-audio blip ~1–2s *after* the
turn is a separate **known/unfixed** issue, P10 — fix reverted by preference),
`MUSETALK_FEED_BURST_S` (1.0 — bursts the first 1s of a turn's audio un-paced so the renderer
isn't starved at turn start; cut lip-start lag ~1.9s→~0.8s), `MUSETALK_END_TAIL_FRAMES` (ease-out
neutral frames after speech), `MUSETALK_SIZE` (512 — shrinking it does NOT cut MuseTalk compute),
`MUSETALK_LEAD_FRAMES` (**14, load-bearing** — lower starves the queue → freeze),
`COSYVOICE_PACE_RATE` (1.3, in the cosyvoice server — caps voice production so it doesn't burst
the shared GPU), `CLIENT_JITTER_BUFFER_MS` (raise only for a remote/WAN viewer),
`WEBRTC_VIDEO_BITRATE_MAX` (caps aiortc's VP8 ceiling so the video fits a WAN link), and
`WEBRTC_ICE_SUBNET` (**`100.64.0.0/10`** = pin WebRTC ICE to the Tailscale interface; fixes the
intermittent remote mic — `0` disables). **Full reference: `WORKFLOW.md` §8.**

## Commands

There is **no build/lint/unit-test suite** — don't invent one. The real commands (3 processes;
`scripts/run.ps1` starts the avatar server + pipeline and propagates the MuseTalk env from `.env`):

```bash
# 1. CosyVoice TTS server (DEFAULT = vLLM in WSL, TTFB ~1.1s) — its OWN repo E:\Claude\cosyvoice-local-tts
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"   # serves :8001 in WSL
#    Then set .env COSYVOICE_URL to the WSL IP (NOT localhost — WSL2 relay buffers the stream): `wsl hostname -I`.
#    FALLBACK = Windows PyTorch server (slower, TTFB ~3.4s), set COSYVOICE_URL=http://localhost:8001 :
#      E:\miniconda3\envs\tts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001
#    (COSYVOICE_PACE_RATE defaults to 1.3; the Windows server needs SSL_CERT_FILE=<certifi> —
#     the tts/musetalk conda envs have a broken Windows cert store; see STATUS.md/memory.)

# 2. MuseTalk avatar server — `musetalk` conda env (NOT the pipeline env), serves :8002
E:\miniconda3\envs\musetalk\python.exe -u -m local_services.musetalk_server.app
#    (reads AVATAR_REF / MUSETALK_SIZE / MUSETALK_FPS from the OS env ONLY — no python-dotenv)

# 3. Pipeline — project main env (SYSTEM Python 3.11, has pipecat — NOT a conda env); serves /client
python -m pipeline.main                                             # http://localhost:7860/client/

# --- or start the avatar server + pipeline together ---
.\scripts\run.ps1

# Verify every fragile import resolves WITHOUT keys/network (Pipecat drift check):
python -m scripts.preflight

# Avatar A/V test tooling (headless, no browser; close any /client tab first — server is single-client):
# UNIFIED harness (PREFER THIS): one command = WebRTC probe + pipeline.log parse + offline capture ->
# output/measure_report.json + docs/measure_data.js (docs/workflow-timeline.html auto-uses it on reload).
python -m scripts.measure --offline-capture                        # full turn timeline + handoffs + metrics
#   (the two tools below are what measure.py wraps; run them standalone only for one-off debugging)
python -m scripts._webrtc_probe --mic output/q_ai.wav --lead 8     # drives a turn, records + metrics
E:\miniconda3\envs\musetalk\python.exe -m local_services.musetalk_server._capture output/q_ai.wav  # offline mp4

# Remote-link isolation test (streams a rendered mp4 LIVE as MJPEG, no GPU/WebRTC) — isolate link vs render:
python -m scripts.stream_live
```

`archive/` holds the regression tests kept out of the live tree: `_screech_repro_test.py`
(re-proves the steady-mode screech fix) and `_sync_routing_test.py`.

## Architecture — how one turn flows

`pipeline/main.py` assembles a linear Pipecat `Pipeline`; frames stream through it:

```
mic → transport.input()(+Silero VAD) → STT → aggregator.user()
    → LLM (streamed, sentence-aggregated) → TTS → Avatar → TtfoMeter
    → transport.output() → browser ;  aggregator.assistant() records the bot turn
```

Each stage is built by a thin factory in `pipeline/stages/` from `config` (one
provider, no branching). The whole thing streams: the LLM's first sentence
reaches TTS before the full answer exists, and TTS's first audio chunk reaches
the avatar immediately. `TtfoMeter` (`pipeline/metrics.py`) measures the gap from
`UserStoppedSpeakingFrame` → `BotStartedSpeakingFrame` (the <8 s metric).

**The avatar is a separate GPU process.** `local_services/musetalk_video.py`
(`MuseTalkVideoService`, the pipeline FrameProcessor) ↔ `local_services/musetalk_server/app.py`
(FastAPI ws server, `musetalk` env). Mouth-region lip-sync, no warmup, female portrait via
`AVATAR_REF`, port `:8002`. The wire contract:

- Client → server: a `config` json, `speech_start`/`speech_end`/`reset` json, and
  binary **16 kHz mono PCM** chunks (the TTS audio, resampled client-side).
- Server → client: binary RGB frame buffers at a steady fps, plus
  `video_start`/`video_clock{frames}`/`video_end` markers (counting only *real* rendered frames).

**A/V sync default = `steady`** (video-master): the voice is buffered and released **paced to the
real frames the server reports rendering**, so the voice waits when the render stalls and never
drifts ahead, for a synced start (the user's pick). `live` (audio-master) forwards the voice
immediately (lips best-effort, ~0.75s trail) and is the robust alternative that never pauses. The
client feeds audio to the server REAL-TIME-PACED (`_feed_q`), except the first `MUSETALK_FEED_BURST_S`
(1.0s) of each turn is burst un-paced so the renderer isn't starved at turn start (cut lip-start lag
~1.9s→~0.8s).

**CRITICAL COUPLING (`main.py`):** the per-frame A/V pinning (`sync_with_audio`) is a *no-op* unless
the transport is **non-live** — pipecat 1.3.0 only reads `_video_images` (where tagged frames land)
when `video_out_is_live=False`; with `is_live=True` the tagged frames are silently dropped and video
free-runs. So `video_out_is_live = not config.avatar_sync_with_audio` — never set `is_live`
independently. **One fps everywhere is load-bearing:** the server frame-drop stride, the client
release clock, and `main.py video_out_framerate` must all equal `config.avatar_fps` (MUSETALK_FPS) or
audio/video drift.

## Environment constraints / gotchas (READ before debugging the avatar)

- **MuseTalk: `cudnn.benchmark` MUST stay `False`** (`musetalk_server/app.py`). With it `True`,
  cuDNN re-autotunes on the turn-START segment (different shape than mid-turn) → a **~16s GPU spike
  on the FIRST segment of every turn** → lips start ~5s late + the render falls behind on long
  replies ("audio ends, avatar keeps moving"). `False` removed it (steady-state per-frame time was
  unchanged). See `docs/PROBLEMS-AND-FIXES.md` P1. Diagnose render-stage timing with `MUSETALK_PROFILE=1`.
- **Judging audio garble: use a CONCATENATED WAV, never per-chunk RMS** — chunks aren't
  sample-aligned, so a single chunk reads as "loud garbage" even when the stream is clean (this
  cost hours; see PROBLEMS-AND-FIXES.md P3 method note).
- **onnxruntime / torch CUDA DLLs:** the avatar server adds torch's `lib/` dir to the DLL search
  path before importing onnxruntime, or onnxruntime silently falls back to CPU (~5× too slow,
  laggy/desynced avatar). Keep that.
- **`conda run` buffers stdout** — a running server's log looks empty; use the `-u` env-python
  invocation above for live logs.
- **The avatar server is single-client.** Fully close the browser tab between tries; a watchdog
  logs throughput and surfaces silent worker-thread crashes.
- **Windows console is cp1252** — `main.py` reconfigures stdout to UTF-8 so the Pipecat banner
  doesn't crash startup. Keep `.py` server source ASCII-safe.
- **conda env cert store:** the `musetalk`/`tts` conda envs have a broken Windows cert store (ssl
  ASN1) that kills torch.hub/urllib downloads — fix = curl-cache the weights + set `SSL_CERT_FILE`
  to certifi. See the `project-visualllm-conda-ssl-weights` memory.
- The `/client` UI is the **pipecat prebuilt bundle**, served as-is — don't add UI hacks back. The
  **one** sanctioned injection is the receive-side jitter buffer (`_install_client_jitter_buffer()`
  in `main.py`, gated by `CLIENT_JITTER_BUFFER_MS`): FastAPI middleware injects a `<script>` into the
  served index before the bundle; the bundle itself is untouched. Keep new client behavior to that
  same pattern (env-gated, bundle untouched) rather than forking the prebuilt dist.
- **Open the client at `/client/` WITH the trailing slash** — the prebuilt page references its
  assets relatively, so `/client` (no slash) 404s them → white screen.

## Conventions

- Keep stage factories single-provider and thin; config is `.env`-driven only.
- Comments state the *why* (latency, a Pipecat quirk, a hardware constraint) — match that voice.
- Accepted tradeoffs (see `STATUS.md`): echo-guard defaults OFF (`ECHO_GUARD=0`, barge-in — use
  headphones) because the half-duplex mute (`=1`) is broken under the default `steady` sync (mic
  stuck-muted after a turn, `docs/PROBLEMS-AND-FIXES.md` P11); `=1` is valid only with
  `MUSETALK_SYNC_MODE=live`. On the single shared GPU the
  lips can trail the voice under load in `live` mode — that's the cost of `live` never freezing; the
  SAFE next lever is bounding the avatar server's `out_q`, **never** re-locking the voice (locked
  sync froze it — see STATUS.md).
- Pipecat import paths drift between releases; the fragile ones are isolated to
  `pipeline/stages/*.py`, `pipeline/main.py`, `pipeline/metrics.py`. Run `python -m scripts.preflight`
  after touching them.
- **The user is on RDP** into this box (and views the live avatar remotely from a notebook in
  another country via the `tailscale serve` HTTPS URL). RDP adds its own video choppiness AND
  desyncs audio/video — when judging avatar smoothness/sync, use `_capture.py` (offline, no
  WebRTC/RDP) or a native remote browser, never the RDP window; re-encode any mp4 trims
  (`-ss -c copy` breaks playback). When the avatar "won't show" or "won't talk," first check both
  processes are up (`:7860` and `:8002`) and that the pipeline picked up the latest code (restart
  it; a stale process lacks recent fixes).
