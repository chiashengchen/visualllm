# VisualLLm — Detailed Workflow

A real-time **speech → STT → LLM → TTS → talking-head avatar** system.
You talk, it transcribes you, an LLM answers, the answer is spoken, and a GPU renders a
lip-synced face — all streaming end-to-end over WebRTC to a browser. Target:
time-to-first-output (TTFO) **< 8 s**.

> This document is the *how it works* companion to `STATUS.md` (current state / decisions)
> and `CLAUDE.md` (repo conventions). When in doubt about live state, `STATUS.md` wins.

---

## 1. The big picture — three processes + a browser

The system is **three separate programs** plus the user's browser. They are intentionally
separate: the TTS LLM and the avatar render each need their own GPU environment.

| Process | Runs in | Port | Job |
|---|---|---|---|
| **CosyVoice TTS** (`run_vllm_server.sh`) | `cosyvllm` conda env in **WSL** (GPU/vLLM) | **8001** | Text → streamed voice audio. Separate repo `E:\Claude\cosyvoice-local-tts` |
| **Avatar server** (`local_services/musetalk_server/app.py`) | `musetalk` conda env (GPU) | **8002** | Turns voice audio into lip-synced RGB frames (MuseTalk) |
| **Pipeline** (`pipeline/main.py`) | system Python 3.11 (has Pipecat) | **7860** | VAD→STT→LLM→TTS→avatar glue, serves the web client, owns the WebRTC connection |

The pipeline talks to the avatar over a **local websocket** (`ws://localhost:8002/stream`) and
to CosyVoice over HTTP (`COSYVOICE_URL`). The browser only ever talks to the **pipeline** (7860),
over WebRTC.

```
  ┌─────────────┐        WebRTC         ┌──────────────────────────┐   local ws    ┌────────────────────┐
  │   BROWSER   │  mic up / A+V down    │   PIPELINE  (:7860)      │  16kHz PCM →  │  AVATAR (:8002)    │
  │ (your laptop)│ ◄──────────────────► │  STT→LLM→TTS→avatar glue │  ◄ RGB frames │  MuseTalk GPU render│
  └─────────────┘                       └──────────────────────────┘               └────────────────────┘
```

---

## 2. One turn, end to end

`pipeline/main.py` assembles a linear Pipecat `Pipeline`; frames stream through it. A
single conversational turn flows like this:

```
mic → transport.input() (+ Silero VAD)
    → STT          (Deepgram nova-2)           : audio → text  ("hello")
    → aggregator.user()                        : builds the user message
    → LLM          (OpenRouter)                : streamed, sentence-by-sentence answer
    → TTS          (CosyVoice2 on vLLM, :8001) : text → voice audio chunks
    → Avatar       (MuseTalkVideoService)      : voice → lip-synced video (via :8002)
    → TtfoMeter                                : measures UserStoppedSpeaking → BotStartedSpeaking
    → transport.output()                       : audio + video → browser
;   aggregator.assistant()                     : records the bot turn into context
```

**It all streams.** The LLM's *first sentence* reaches TTS before the full answer is
generated, and TTS's *first audio chunk* reaches the avatar immediately. That overlap is
what keeps the whole thing inside the < 8 s budget.

`TtfoMeter` (`pipeline/metrics.py`) logs `[TTFO]` per turn — the gap from
`UserStoppedSpeakingFrame` to `BotStartedSpeakingFrame`.

---

## 3. Stage-by-stage detail

Each stage is built by a **thin, single-provider factory** in `pipeline/stages/`, driven
only by `.env` (no provider-selection branching — see `CLAUDE.md`).

### VAD — Silero (local)
`pipeline/stages/vad.py` (`build_vad_params`). Runs on the input audio to detect when you
**stop** speaking, which ends the user turn and kicks STT's final transcript.

### STT — Deepgram nova-2
`pipeline/stages/stt.py`. Streams your mic audio to Deepgram over a websocket; emits
interim + final transcripts. Language follows `LANGUAGE` (`en-US` / `zh-TW` / `th`). Needs
`DEEPGRAM_API_KEY`.

### LLM — OpenRouter
`pipeline/stages/llm.py`. Any model via `OPENROUTER_MODEL` (default Gemini Flash Lite). The
response is **streamed and sentence-aggregated** so TTS can start on sentence 1. A short
system prompt (in `pipeline/config.py`) keeps replies spoken-style (no markdown/emoji).
Needs `OPENROUTER_API_KEY`.

### TTS — CosyVoice2 on vLLM (default) / ElevenLabs / Deepgram Aura (fallbacks)
`pipeline/stages/tts.py`. Converts the LLM text to voice audio chunks, streamed.
- **Default**: `TTS_PROVIDER=cosyvoice` → the local CosyVoice2-0.5B server at `COSYVOICE_URL`
  (female zero-shot voice, covers en/zh). It runs its autoregressive LLM on **vLLM in WSL** →
  first-chunk latency ~3.4s → **~1.1s** (separate repo `E:\Claude\cosyvoice-local-tts`,
  `run_vllm_server.sh`). **`COSYVOICE_URL` must be the WSL IP, NOT localhost** (WSL2's localhost
  relay buffers the streaming audio ~2s). The original Windows `tts`-env PyTorch server is the
  fallback (set `COSYVOICE_URL=http://localhost:8001` + start it).
- **Fallbacks**: `TTS_PROVIDER=elevenlabs` (`flash_v2_5`, multilingual cloud) or `deepgram`
  (Deepgram **Aura**, reuses `DEEPGRAM_API_KEY`, English-only).

### Avatar (client side) — `MuseTalkVideoService`
`local_services/musetalk_video.py`, a Pipecat `FrameProcessor` sitting between TTS and the
transport. It:
1. Resamples each TTS audio chunk to **16 kHz mono PCM** and feeds it to the avatar server
   (the first `MUSETALK_FEED_BURST_S` of a turn un-paced, then real-time-paced so no backlog).
2. **Buffers the downstream voice copy** and releases it frame-clocked to the real video
   the server reports rendering (so audio never runs ahead of the lips), tagging frames for
   A/V sync in `steady` mode (see §5).
3. Keeps every downstream audio frame whole-sample (`_align_even`) — the anti-screech guard.

### Avatar (server side) — MuseTalk on the GPU
`local_services/musetalk_server/app.py` (FastAPI websocket). See §4.

### TtfoMeter
`pipeline/metrics.py`. Pure measurement — logs the < 8 s metric per turn and a summary on
disconnect.

---

## 4. The avatar subsystem in depth

### The wire contract (client ↔ server)
To understand either side you must read both (`musetalk_video.py` + `musetalk_server/app.py`):

**Client → server:**
- `{"type":"config","fps":20}` — sets output fps.
- `{"type":"speech_start"}` / `{"type":"speech_end"}` / `{"type":"reset"}` — turn markers / barge-in.
- binary **16 kHz mono PCM** chunks (the TTS audio).

**Server → client:**
- binary RGB frame buffers at a steady fps.
- `{"type":"video_start"}` / `{"type":"video_clock","frames":N}` / `{"type":"video_end"}` —
  sync markers counting only *real* rendered frames; these clock the client's voice release.

### Inside the server
- **Mouth-region lip-sync.** MuseTalk animates the mouth region of the `AVATAR_REF` portrait —
  no diffusion warmup, sharp lips. A `pump()` drains rendered frames to the websocket at a steady
  fps, showing an idle/neutral frame between turns.
- **Single-client + session guard.** The model is shared and single-client; a session guard
  serializes whole sessions so a reconnect can't re-init over live worker threads. A watchdog logs
  throughput and surfaces silent worker-thread crashes.
- **`cudnn.benchmark = False` (load-bearing).** With it `True`, cuDNN re-autotunes on the turn-START
  segment (a different shape than mid-turn) → a ~16s GPU spike on the first segment of every turn →
  lips start ~5s late. `False` removes it (steady-state per-frame time unchanged). See
  `docs/PROBLEMS-AND-FIXES.md` P1; diagnose render timing with `MUSETALK_PROFILE=1`.
- **Frame count = `audio_sec × fps` (keep `MUSETALK_FPS` a divisor of 16000).** Each render segment
  is sized by `MuseTalkEngine.samples_for_frames(n)=ceil(n*16000/fps)` so the renderer
  (`floor(len/sr*fps)`) yields exactly `SEG_FRAMES` per batch. The old `int(16000/fps)*SEG_FRAMES`
  sizing truncated at fps that don't divide 16000 (e.g. 12 → 7 frames/8-frame batch), losing ~12.5%
  of frames over a turn so the lips finished ~1–2s before the voice. See `docs/PROBLEMS-AND-FIXES.md`
  P9. (A leftover-audio blip ~1–2s *after* the turn is a separate, **known + unfixed** issue — P10,
  fix reverted by preference.)
- **OS env only.** The server reads `AVATAR_REF` / `MUSETALK_SIZE` / `MUSETALK_FPS` from the OS
  environment (no `python-dotenv` in its conda env); `scripts/run.ps1` propagates them from `.env`.

---

## 5. A/V synchronization (how lips stay on the voice)

The HARD constraint: **MuseTalk and CosyVoice share ONE GPU.** MuseTalk renders ~20 fps alone but
CosyVoice bursts the GPU while streaming a reply and slows the render unpredictably. Two modes
(`MUSETALK_SYNC_MODE`):

- **steady (DEFAULT) — VIDEO-MASTER:** the voice is buffered and released paced to the server's
  `video_clock` markers (real rendered frames), so the voice waits when the render stalls and never
  drifts ahead — a synced start. Per-frame pinning: each turn's frames are tagged
  `OutputImageRawFrame.sync_with_audio=True` and routed through the transport's **audio queue** so
  each frame displays at its audio position. **Load-bearing coupling (`main.py`):** this only works
  when the transport is **non-live**, so `video_out_is_live = not config.avatar_sync_with_audio`.
  With `is_live=True`, tagged frames are silently dropped and video free-runs (drifts). Remaining
  tradeoff: under a long render stall the voice briefly **pauses** then resumes clean.
- **live — AUDIO-MASTER:** the voice is forwarded immediately so it **can never freeze**; the lips
  are best-effort (~0.75s trail under contention).

**One fps everywhere.** The server frame-drop stride, the client release clock, and
`main.py video_out_framerate` must all equal **`MUSETALK_FPS`** or audio/video drift.

**The screech (fixed).** In steady, a >3s render stall used to starve pipecat's audio queue → its
3s `_bot_stopped_speaking` timeout fired mid-turn → it discarded the odd partial audio buffer →
odd-byte misalignment = screech. Fixed two ways: `main.py::_relax_bot_vad_stop_timeout()` raises the
timeout (we drive an explicit `TTSStoppedFrame` per turn anyway) AND `musetalk_video.py::_align_even`
keeps the downstream PCM whole-sample. See `docs/PROBLEMS-AND-FIXES.md` P3.

---

## 6. Running the system (locally, on the GPU box)

```bash
# 1. CosyVoice TTS server — vLLM in WSL (TTFB ~1.1s). Then set COSYVOICE_URL to the WSL IP.
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"   # :8001
#   vLLM shares the 16GB card with MuseTalk. gpu_memory_utilization (COSYVOICE_VLLM_GPU_UTIL,
#   default 0.3, in the cosyvoice repo) must be high enough for vLLM's ~4GB footprint or it
#   crashes "No available memory for the cache blocks" (0.2 was too low). Raise toward 0.35 if a
#   heavy GPU app is closed; lower if MuseTalk OOMs. The log line "Available KV cache memory" must
#   be positive.

# 2 + 3. MuseTalk avatar server + pipeline (one script: starts both, propagates the MuseTalk knobs)
.\scripts\run.ps1
#   (or run them by hand:)
#   E:\miniconda3\envs\musetalk\python.exe -u -m local_services.musetalk_server.app    # :8002
#   python -m pipeline.main                                                            # :7860 → /client
```

Then open `http://localhost:7860/client/` (**trailing slash**), allow the mic, use **headphones**
(echo-guard is on; `ECHO_GUARD=0` to allow barge-in), and talk. Fully close the tab between tries
(sessions are serialized server-side).

**Sanity check without keys/network** (catches Pipecat import drift):
```bash
python -m scripts.preflight
```

---

## 7. Remote viewing (across the internet)

The avatar is WebRTC, so you view it natively in a remote browser — **never over RDP**
(RDP carries audio and video on separate paths and desyncs them; judge sync only natively).

The box runs **Tailscale Serve**, which proxies the pipeline over HTTPS on the tailnet:

```
https://porsche-pc.tail21bb8a.ts.net/client/
```

- **HTTPS** is required so the browser allows the **microphone** (a plain `http://<ip>` LAN
  URL blocks the mic — insecure origin).
- Works from any device logged into the same Tailscale account.
- The pipeline binds to localhost only; Tailscale Serve proxies locally, so that's fine.
- If the remote mic is flaky ("works sometimes, mostly not"), that's **WebRTC ICE candidate
  pollution** — `WEBRTC_ICE_SUBNET=100.64.0.0/10` pins ICE to the Tailscale interface.

**Network reality:** over a cross-border link (e.g. Taiwan ↔ Thailand) there is real jitter.
Audio (~30 kbps) sails through; the video is heavier. The fix is to make the stream **fit the link**,
in order of effectiveness:
1. **Fit the stream to the link** (the real fix) — small frame (`MUSETALK_SIZE`, e.g. 320) + a
   bounded send bitrate (`WEBRTC_VIDEO_BITRATE_MAX`, ~600k) so the video can't starve the link;
   then a *small* jitter buffer (`CLIENT_JITTER_BUFFER_MS`) absorbs the leftover timing variance.
   Counter-intuitively, bitrate that is too LOW starves the VP8 encoder → choppier — don't
   shrink-and-starve.
2. **Receive-side jitter buffer** alone — `CLIENT_JITTER_BUFFER_MS`. Smooths jitter at the cost of
   latency; too high makes the avatar trail. After changing it, **hard-refresh** the browser
   (Ctrl+Shift+R) — the page is cached.
3. **Isolate link vs render** — `scripts/stream_live.py` streams a rendered mp4 LIVE as MJPEG (no
   GPU/WebRTC) so you can tell a link problem from a render problem.

---

## 8. Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LANGUAGE` | `en` | `en` / `zh` / `th` (STT + voice) |
| `ECHO_GUARD` | `1` | mute the mic while the bot speaks (half-duplex); `0` allows barge-in |
| `TTS_PROVIDER` | `cosyvoice` | `elevenlabs` (multilingual cloud) / `deepgram` (Aura, en-only) fallbacks |
| `COSYVOICE_URL` | `http://localhost:8001` | the CosyVoice server — set to the **WSL IP** for the vLLM server (NOT localhost; the relay buffers the stream), localhost for the Windows fallback |
| `COSYVOICE_VOICE` / `COSYVOICE_PACE_RATE` | `weather` / `1.3` | zero-shot speaker id / GPU-pacing cap (server-side) |
| `DEEPGRAM_TTS_VOICE` | `aura-2-helena-en` | Aura voice when `TTS_PROVIDER=deepgram` |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash-lite` | any OpenRouter model |
| `AVATAR_REF` | `assets/avatar_female.png` | portrait (image or video) the MuseTalk server animates |
| `MUSETALK_SYNC_MODE` | `steady` | video-master, synced start (user's pick + default). The old steady "screech" is FIXED (`_align_even` whole-sample guard + `BOT_VAD_STOP_FALLBACK_SECS` raise, `docs/PROBLEMS-AND-FIXES.md` P3). Tradeoff: under a long render stall the voice briefly pauses then resumes clean. `live` = audio-master (voice never pauses, lips trail ~0.75s) is the alternative |
| `MUSETALK_FPS` / `MUSETALK_SIZE` | `20` / `512` | avatar output fps / frame px (shrinking SIZE does NOT cut MuseTalk compute). **Keep FPS a divisor of 16000** (8/10/16/20/25) so frame count = audio length; the `samples_for_frames` fix makes the current `12` correct too (P9) |
| `COSYVOICE_VLLM_GPU_UTIL` | `0.3` | *(set in the **cosyvoice repo**, not this `.env`)* fraction of the 16GB card vLLM may use. 0.2 was too low (< its ~4GB footprint → KV cache crash); 0.3 fits alongside MuseTalk. Raise to ~0.35 with more free VRAM |
| `MUSETALK_LEAD_FRAMES` | `14` | video-start cushion — **load-bearing** (lower starves the queue → freeze) |
| `MUSETALK_FEED_BURST_S` | `1.0` | burst the first 1s of a turn's audio un-paced → renderer not starved at turn start (lip-start lag ~1.9s→~0.8s; `docs/PROBLEMS-AND-FIXES.md` P2) |
| `MUSETALK_END_TAIL_FRAMES` | `10` | ease-out neutral frames after speech (softer mouth-close) |
| `BOT_VAD_STOP_FALLBACK_SECS` | `600` | steady-screech fix: keep high so a render stall can't discard the partial audio buffer |
| `WEBRTC_ICE_SUBNET` | `100.64.0.0/10` | pin WebRTC ICE host candidates to the Tailscale interface (fixes the intermittent remote mic); `0` = advertise all |
| `CLIENT_JITTER_BUFFER_MS` | `400` | receive-side WebRTC jitter buffer (0 = off); raise for a remote viewer |
| `WEBRTC_VIDEO_BITRATE_MAX` | `600000` | VP8 send-bitrate ceiling, bits/s (0 = aiortc default 1.5M) |
| `TTFO_TARGET_SECONDS` | `8` | the < 8 s target for logging |

Keys required: `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY` (and `ELEVENLABS_API_KEY` +
`ELEVENLABS_VOICE_ID` only if `TTS_PROVIDER=elevenlabs`).

---

## 9. Key files

| File | Role |
|---|---|
| `pipeline/main.py` | Pipeline assembly, transport params, greeting, A/V-sync coupling, the screech fix + the three remote-WebRTC fixes |
| `pipeline/config.py` | All `.env`-driven config + system prompts |
| `pipeline/stages/*.py` | Per-stage single-provider factories (vad/stt/llm/tts/avatar) |
| `pipeline/metrics.py` | `TtfoMeter` (the < 8 s metric) |
| `local_services/musetalk_video.py` | Client-side avatar processor + frame-clocked A/V sync + `_align_even` |
| `local_services/musetalk_server/app.py` | MuseTalk GPU server (ws, frame pump, sync markers, session guard) |
| `local_services/cosyvoice_tts.py` | CosyVoice streaming TTS client |
| `scripts/preflight.py` | Import/drift check |
| `STATUS.md` | Live state + decision log (source of truth) |

---

## 10. Troubleshooting (failure modes we've actually hit)

| Symptom | Cause | Fix |
|---|---|---|
| Avatar shows but **won't talk** (voice + chat) | TTS server down — most often CosyVoice crashed on the **shared GPU** (vLLM "No available memory for the cache blocks") | Check `:8001` is up (`wsl ... ss -ltn`); if it crashed, free VRAM and/or raise `COSYVOICE_VLLM_GPU_UTIL`. The pipeline log shows "Cannot connect to host …:8001". Or swap `TTS_PROVIDER` |
| Lips **finish ~1–2s before the voice** on long replies | per-segment frame deficit when `MUSETALK_FPS` doesn't divide 16000 | Fixed (P9, `samples_for_frames`); keep FPS a divisor of 16000. Verify warmup logs `8 frames/segment`, not 7 |
| **Leftover-audio blip** ~1–2s after the turn ends | steady's floor-cap strands the final sub-frame to the delayed `video_end` drain | **Known issue, fix reverted by preference** (P10). Re-apply = `int()`→`ceil()` in `musetalk_video.py::_advance` |
| **Avatar not showing** at all | MuseTalk server (:8002) down | Start the avatar server (the pipeline needs it) |
| Lips start **~5s late** + render falls behind on long replies | `cudnn.benchmark=True` (per-turn re-autotune spike) | Keep it `False` in `musetalk_server/app.py` |
| Voice **screeches** mid-reply in steady mode | pipecat discarding the odd partial audio buffer on a render-stall gap | Keep `BOT_VAD_STOP_FALLBACK_SECS` high + `_align_even` (both already in) |
| Lips **drift / out of sync** in browser | `video_out_is_live=True` dropping synced frames | Use `steady` mode (couples is_live off); one `MUSETALK_FPS` everywhere |
| Video **lags / stutters** remotely, audio fine | oversized WebRTC stream on a WAN link | Fit the stream: smaller `MUSETALK_SIZE` + `WEBRTC_VIDEO_BITRATE_MAX`; tune `CLIENT_JITTER_BUFFER_MS` |
| **Remote mic dies** mid-call ("works sometimes") | WebRTC ICE candidate pollution | `WEBRTC_ICE_SUBNET=100.64.0.0/10` pins ICE to Tailscale |
| Avatar **laggy on the GPU box itself** | onnxruntime fell back to CPU, or fps mismatch | Verify CUDA DLLs on path; keep one `MUSETALK_FPS` everywhere |
| Judging sync **over RDP** looks wrong | RDP desyncs audio/video paths | Judge natively (remote browser) or via `_capture.py` offline |
