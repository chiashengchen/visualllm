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

**Current default stack (2026-06-22 — fully local TTS + avatar). See `STATUS.md` for the full
state + the A/V-sync architecture decision (read it before touching sync).**

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en-US`/`zh-TW` by `LANGUAGE`) | cloud |
| LLM | OpenRouter (any model via `OPENROUTER_MODEL`) | cloud |
| TTS | **CosyVoice2-0.5B** local streaming server (female zero-shot voice), **now on vLLM in WSL** (TTFB 3.4s→~1.1s; the Windows `tts`-env PyTorch server is the fallback) | **`:8001`, separate repo `E:\Claude\cosyvoice-local-tts`** |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait) | **`:8002`, `musetalk` conda env** |

**TTS note (2026-06-22):** CosyVoice now runs its autoregressive LLM on **vLLM inside WSL Ubuntu**
(`cosyvllm` conda env on the Blackwell 5060 Ti) — this cut first-chunk latency ~3.4s→~1.1s, the root
cause of the avatar lip-lag. The pipeline reaches it via `COSYVOICE_URL` set to the **WSL IP**, NOT
`localhost` (WSL2's localhost relay buffers the streaming audio ~2s). Run it with
`bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh` in WSL. The original Windows `tts`-env
PyTorch server is the fallback (set `COSYVOICE_URL=http://localhost:8001` + start it). Full build
notes + gotchas: the `project-visualllm-cosyvoice-vllm` memory.

Each stage is still a thin single-provider factory in `pipeline/stages/` chosen by `.env` — these
are **deliberate fallback switches, not multi-provider branching**:
- `TTS_PROVIDER` = `cosyvoice` (default) | `elevenlabs` | `deepgram` (the last two are cloud fallbacks).
- `AVATAR` = `musetalk` (default) | `ditto` (the full-face TensorRT path, kept) | `none` (audio-only, client renders the face).

Core `.env` knobs: `LANGUAGE` (en/zh/th), `TTFO_TARGET_SECONDS`, `AVATAR`, `TTS_PROVIDER`,
`MUSETALK_SYNC_MODE` (**`steady`** = video-master, synced start, the user's pick and current
default; `live` = audio-master, voice instant + lips trail ~0.75s, can never pause. The old
**steady "screech" is FIXED** — it was pipecat discarding the partial audio buffer after a >3s
render-stall gap (`BOT_VAD_STOP_FALLBACK_SECS`), see `docs/PROBLEMS-AND-FIXES.md` P3 +
`main.py::_relax_bot_vad_stop_timeout`. Remaining steady tradeoff: under a long render stall the
voice briefly **pauses** then resumes clean — switch to `live` if that pause is worse than the lip
trail), `MUSETALK_FPS` (12),
`MUSETALK_FEED_BURST_S` (1.0 — bursts the first 1s of a turn's audio un-paced so the renderer
isn't starved at turn start; cut lip-start lag ~1.9s→~0.8s), `MUSETALK_END_TAIL_FRAMES` (ease-out
neutral frames after speech), `MUSETALK_SIZE` (256 — shrinking it does NOT cut MuseTalk compute),
`MUSETALK_LEAD_FRAMES` (**14, load-bearing** — lower starves the queue → freeze),
`COSYVOICE_PACE_RATE` (1.3, in the cosyvoice server — caps voice production so it doesn't burst
the shared GPU), `CLIENT_JITTER_BUFFER_MS` (150 local; raise only for a remote/WAN viewer),
`WEBRTC_ICE_SUBNET` (**`100.64.0.0/10`** = pin WebRTC ICE to the Tailscale interface; fixes the
intermittent remote mic — `0` disables, see STATUS.md 2026-06-22), and the `DITTO_*` knobs when
`AVATAR=ditto`. **Full reference: `WORKFLOW.md` §8.**

## Commands

There is **no build/lint/unit-test suite** — don't invent one. The real commands (DEFAULT stack —
3 processes; `scripts/run.ps1` starts #2+#3 and propagates the avatar env from `.env`):

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

# --- or start the avatar server + pipeline together (picks the engine from AVATAR in .env) ---
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

# --- FALLBACK avatar: Ditto (AVATAR=ditto), `ditto` conda env, also :8002 ---
conda run -n ditto python -m local_services.ditto_server.app
E:\miniconda3\envs\ditto\python.exe -u -m local_services.ditto_server.app   # live logs (conda run buffers)
conda run -n ditto python -m local_services.ditto_server.ws_test ws://localhost:8002/stream

# --- Avatar perf/sync tooling (run in the `ditto` env; server must be free of other
#     clients — STOP the pipeline first, the server is single-client) ---
# Build the Blackwell FP16 TensorRT engines (ONE-TIME per machine; see DITTO_TRT below):
#   build_trt.py builds decoder/stitch/appearance (ORT-validated). The WARP engine +
#   its GridSample3D plugin are built separately (need the compiler) under plugin_build/:
E:\miniconda3\envs\ditto\python.exe -m local_services.ditto_server.build_trt
#   Rebuild the warp plugin DLL (sm_120/TRT10) then the warp engine (ONE-TIME per GPU/TRT):
cmd /c local_services\ditto_server\plugin_build\_build_plugin.bat   # -> grid_sample_3d_plugin.dll
#   (then build warp_network_fp16.engine with the plugin loaded; see STATUS.md perf round 3)
# Record the LIVE real-time stream to an mp4 (judge the avatar WITHOUT WebRTC/RDP;
#   --lead S delays video by S to test/fix lip offset):
E:\miniconda3\envs\ditto\python.exe -m local_services.ditto_server.capture_mp4 \
    --wav output/what_is_ai.wav --out output/realtime_capture.mp4 --fps 12
# Sustained render fps under a long turn (real_fps, not pump-padded):
E:\miniconda3\envs\ditto\python.exe -m local_services.ditto_server._fps_probe ws://localhost:8002/stream 6 12
# Authoritative lip-vs-audio offset on RAW frames (mouth-motion xcorr audio-RMS):
E:\miniconda3\envs\ditto\python.exe -m scripts.avatar_tune align --set "FPS=12,OVERLAP=25,STEPS=25,TRT=1"
```

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

**The avatar is a separate GPU process.** Two engines share ONE wire contract + port (`:8002`),
chosen by `AVATAR`:
- **MuseTalk (default):** `local_services/musetalk_video.py` (`MuseTalkVideoService`, the pipeline
  FrameProcessor) ↔ `local_services/musetalk_server/app.py` (FastAPI ws server, `musetalk` env).
  Mouth-region lip-sync, no warmup, female portrait via `AVATAR_REF`. **A/V sync default = `steady`
  (video-master): the voice is held and released locked to rendered frames for a synced start (the
  user's pick).** `live` (audio-master) forwards the voice immediately (lips best-effort, ~0.75s
  trail) and is the robust alternative. The client feeds audio to the server REAL-TIME-PACED
  (`_feed_q`), except the first `MUSETALK_FEED_BURST_S` (1.0s) of each turn is burst un-paced so the
  renderer isn't starved at turn start (cut lip-start lag ~1.9s→~0.8s). **The steady "screech" is
  FIXED** (was pipecat discarding the partial audio buffer after a >3s render-stall gap →
  odd-byte misalignment; `main.py::_relax_bot_vad_stop_timeout` raises `BOT_VAD_STOP_FALLBACK_SECS`).
  Steady's only remaining tradeoff: under a long render stall the voice briefly pauses then resumes
  clean — use `live` if that's worse than the lip trail. See `docs/PROBLEMS-AND-FIXES.md` P1–P3, P7.
- **Ditto (fallback, `AVATAR=ditto`):** the full-face TensorRT path below.

The Ditto detail that follows describes the **fallback** engine. Ditto and MuseTalk
(`DittoVideoService` / `MuseTalkVideoService`) share this wire contract — to understand either, read both:

- Client → server: a `config` json, `speech_start`/`speech_end`/`reset` json, and
  binary **16 kHz mono PCM** chunks (the TTS audio, resampled client-side).
- Server → client: binary **512×512×3 RGB** frame buffers at a steady fps.
- Crucially, `DittoVideoService` streams the TTS audio to the server immediately
  (rendering can't wait) but **frame-clocks the downstream copy**: the voice is
  buffered and released paced to the real frames the server reports rendering
  (`video_start`/`video_clock`/`video_end` markers), so the voice waits when the
  render stalls and never drifts ahead. This is why any A/V-sync work belongs in
  `ditto_video.py` — tune `DITTO_SYNC_LEAD_S` for the lip lead.

Inside the server, Ditto's `StreamSDK` normally writes frames to an mp4; we
**subclass it and override `_writer_worker`** to divert finished frames onto a
queue that the websocket `pump` drains at a steady fps (idle/neutral frame
between turns). The SDK is single-client and shared; `_session_lock` serializes
whole sessions so a reconnect can't re-`setup()` over live worker threads.

**TensorRT acceleration (`DITTO_TRT`, default on).** The GPU is Blackwell (sm_120);
the photoreal render was the bottleneck. `build_trt.py` compiles FP16 TensorRT engines
**for this card** from Ditto's ONNX graphs (numerically validated vs fp32),
`trt_runner.py` runs them via torch tensors (no `cuda-python`), and
`app.py::_enable_trt` swaps **decoder/stitch/appearance/warp** to engines (diffusion +
aux stay PyTorch). **warp** also runs as a TRT engine via a self-built **GridSample3D
plugin** (`grid_sample_3d_plugin.dll`, registered by `_load_grid_sample_plugin()`;
gated — if the .dll is absent warp falls back to PyTorch). Net per-frame on this card:
decode ~24ms, warp 47→~26ms; **render ~14 → ~26 real fps (full native 25 fps)**. Engines
+ plugin are GPU-arch-specific, cached in `checkpoints/ditto_trt_blackwell/`; rebuild on a
new GPU. `DITTO_TRT=0` = pure-PyTorch fallback (then drop `DITTO_FPS` to 8). Two
**lossless no-TRT** wins also landed: putback composites only the face bbox in float32
(75→~36ms, the old hidden CPU gate), and `DITTO_PROFILE` (default off) logs per-stage ms.
Since the renderer now beats native 25, `DITTO_FPS` can be raised (12 → ~20-25) for a
smoother avatar — pending the live lip-sync check.

**A/V sync — two mechanisms, both in `ditto_video.py`.** (1) Frame-clocked voice
release (paced to `video_clock` markers). (2) `sync_with_audio` mode
(`DITTO_SYNC_WITH_AUDIO`, default on): each turn's frames are buffered and pushed
tagged `OutputImageRawFrame.sync_with_audio=True` right after their matching voice, so
the transport shows each frame at its audio position instead of an independent video
clock. **CRITICAL COUPLING (`main.py`):** `sync_with_audio` is a *no-op* unless the
transport is **non-live** — pipecat 1.3.0 only reads `_video_images` (where tagged
frames land) when `video_out_is_live=False`; with `is_live=True` the tagged frames are
silently dropped and video free-runs (the long-standing live desync). So
`video_out_is_live = want_video and not DITTO_SYNC_WITH_AUDIO` — never set `is_live`
independently. **One fps everywhere is load-bearing:** the server frame-drop stride, the
client release clock, and `main.py video_out_framerate` must all equal `DITTO_FPS`
(default 12) or audio/video drift. The remaining felt "delay" is Ditto's **diffusion
warmup** (`valid_clip_len = 80 - DITTO_OVERLAP` of audio before the first frame, ~2.2s at
overlap 25) — masked by a captured idle loop (`DITTO_IDLE_CAPTURE_CHUNKS`), NOT removable
without breaking lip-sync (higher overlap/fewer steps destabilize it; 25/25 is the sweet spot).

**Remote viewing + jitter buffer.** The avatar is WebRTC, so view it **natively** in a
remote browser, never over RDP (RDP carries audio/video on separate paths and desyncs
them). `tailscale serve` exposes the pipeline over HTTPS (needed so the browser allows the
mic) at `https://<machine>.<tailnet>.ts.net/client`. Over a WAN, audio (~30 kbps) is fine
but the 512px video (~1-3 Mbps) hits jitter. `pipeline/main.py::_install_client_jitter_buffer()`
injects a receive-side jitter buffer into the served `/client` page (every device, no
console tweak) via `CLIENT_JITTER_BUFFER_MS`. Tradeoff: too low → stutter, too high → the
avatar trails the voice. After changing it, **hard-refresh** the browser (the page is cached).
Open the client at **`/client/` with the trailing slash** — the prebuilt page references its
assets relatively, so `/client` (no slash) 404s them → white screen.

**The bigger remote levers are frame size + bitrate, not the buffer (STATUS 2026-06-20).**
`DITTO_SIZE` (smaller frame) and `WEBRTC_VIDEO_BITRATE_MAX` (`main.py::_configure_webrtc_video_bitrate()`
caps aiortc's VP8 ceiling, default on at 600k) fit the stream to the link; the buffer is then
just a latency↔smoothness trim. Counter-intuitively **bandwidth is usually NOT the hard limit** —
starving the bitrate too low (e.g. 200k) makes the VP8 encoder drop frames → *choppier*, so don't
shrink-and-starve. Isolate link-vs-render with **`scripts/stream_live.py`** (streams a rendered mp4
LIVE as MJPEG over `tailscale serve --set-path /watch`, no GPU/WebRTC) before tuning. **NOTE:** the
Ditto server reads `DITTO_SIZE` from the OS env only (no `python-dotenv` in the `ditto` env), so set
it for *both* processes — `scripts/run.ps1` now propagates it from `.env` to both children.

## Environment constraints / gotchas (READ before debugging the avatar)

- **MuseTalk: `cudnn.benchmark` MUST stay `False`** (`musetalk_server/app.py`). With it `True`,
  cuDNN re-autotunes on the turn-START segment (different shape than mid-turn) → a **~16s GPU spike
  on the FIRST segment of every turn** → lips start ~5s late + the render falls behind on long
  replies ("audio ends, avatar keeps moving"). `False` removed it (steady-state per-frame time was
  unchanged). See `docs/PROBLEMS-AND-FIXES.md` P1. Diagnose render-stage timing with `MUSETALK_PROFILE=1`.
- **Judging audio garble: use a CONCATENATED WAV, never per-chunk RMS** — chunks aren't
  sample-aligned, so a single chunk reads as "loud garbage" even when the stream is clean (this
  cost hours; see PROBLEMS-AND-FIXES.md P3 method note). Env-gated WAV captures exist:
  `MUSETALK_DOWNSTREAM_CAPTURE`, `COSYVOICE_DELIVERED_CAPTURE`, `COSYVOICE_HANDLE_AUDIO_PROBE`,
  `COSYVOICE_CAPTURE_ALL` (all default OFF) → `output/cosy_noise/`.
- **Compiler toolchain IS now installed** (2026-06-18: VS Build Tools MSVC 14.44 + Windows
  SDK 10.0.26100 + CUDA Toolkit 12.8.93; **driver kept at 591.44** — toolkit-only install).
  The two no-compiler workarounds below still stand (harmless, keep them):
  - Ditto's Cython blend kernel (`ditto_server/vendor/.../core/utils/blend/__init__.py`)
    was replaced with a NumPy equivalent. Don't revert to `pyximport`. (Note: putback no
    longer blends the whole frame — it does a float32 bbox composite; see `putback.py`.)
  - **onnxruntime-gpu needs CUDA-12 DLLs on the search path or it silently falls
    back to CPU** (→ ~5× too slow, laggy/desynced avatar). `ditto_server/app.py`
    adds torch's `lib/` dir via `os.add_dll_directory` *before* onnxruntime
    imports — and now also the `tensorrt_libs` dir (for `nvinfer_10.dll`). Keep both.
  - **TensorRT must be 10.x** (`tensorrt-cu12<11`): TRT 11 dropped `BuilderFlag.FP16`
    and onnxruntime needs `nvinfer_10.dll`. The 10.x wheels ship the sm_120 builder
    resource, so engines build for Blackwell with no compiler.
  - **Part B is DONE** (was: install a compiler to TRT the warp stage). The GridSample3D
    TRT plugin is built for Windows/TRT-10.16/sm_120 — source + headers + synthesized
    `nvinfer.lib` live in `local_services/ditto_server/plugin_build/`; the built
    `grid_sample_3d_plugin.dll` is staged in `checkpoints/ditto_trt_blackwell/`. To rebuild
    (new GPU/TRT): see the plugin-build command below. See `STATUS.md` perf round 3.
- **`conda run` buffers stdout** — a running server's log looks empty; use the
  `-u` env-python invocation above for live logs.
- **The Ditto server is single-client.** Fully close the browser tab between
  tries; a watchdog logs throughput and surfaces silent worker-thread crashes
  (the SDK hides worker exceptions until `close()`).
- **Windows console is cp1252** — `main.py` reconfigures stdout to UTF-8 so the
  Pipecat banner doesn't crash startup. Keep `.py` server source ASCII-safe.
- The `/client` UI is the **pipecat prebuilt bundle**, served as-is — don't add UI
  hacks back. The **one** sanctioned injection is the receive-side jitter buffer
  (`_install_client_jitter_buffer()` in `main.py`, gated by `CLIENT_JITTER_BUFFER_MS`):
  FastAPI middleware injects a `<script>` into the served index before the bundle; the
  bundle itself is untouched. Keep new client behavior to that same pattern (env-gated,
  bundle untouched) rather than forking the prebuilt dist.

## Conventions

- Keep stage factories single-provider and thin; config is `.env`-driven only.
- Comments state the *why* (latency, a Pipecat quirk, a hardware constraint) —
  match that voice.
- Accepted tradeoffs (see `STATUS.md`): echo-guard removed (avatar self-talk on
  speakers — use headphones), Azure zh path removed. Avatar is now **MuseTalk** by choice
  (female portrait, no warmup, `local_services/musetalk_server/` + `musetalk_video.py`);
  **Ditto stays in-repo** as the full-face TensorRT fallback (`AVATAR=ditto`), not deleted.
  Accepted A/V tradeoff: on the single shared GPU the lips can trail the voice under load —
  that's the cost of `live` mode never freezing; the SAFE next lever is bounding the avatar
  server's `out_q`, **never** re-locking the voice (locked sync froze it — see STATUS.md).
- Pipecat import paths drift between releases; the fragile ones are isolated to
  `pipeline/stages/*.py`, `pipeline/main.py`, `pipeline/metrics.py`.
- **The user is on RDP** into this box (and views the live avatar remotely from a
  notebook in another country via the `tailscale serve` HTTPS URL). RDP adds its own
  video choppiness AND desyncs audio/video — when judging avatar smoothness/sync, use
  `capture_mp4.py` (offline, no WebRTC/RDP) or a native remote browser, never the RDP
  window; re-encode any mp4 trims (`-ss -c copy` breaks playback). The box runs the
  pipeline + avatar server as long-lived background processes — when the avatar "won't
  show" or "won't talk," first check both are actually up (`:7860` and `:8002`) and that
  the pipeline picked up the latest code (restart it; a stale process lacks recent fixes).
