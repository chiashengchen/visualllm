# VisualLLm — Project Status & Next Steps

_Last updated: 2026-06-23 (avatar frame-deficit fix; baseline `cd88f20`)_

> **See `WORKFLOW.md`** for the full end-to-end system workflow (the processes, the turn
> flow, the avatar wire contract, running locally + remote, config reference).
> **See `docs/PROBLEMS-AND-FIXES.md`** for the catalogue of bugs found + how each was fixed.

## The stack

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en`/`zh`/`th` by `LANGUAGE`) | cloud |
| LLM | OpenRouter (`OPENROUTER_MODEL`) **default**, or a Chinese weather bot via `LLM_PROVIDER=weather_chain` | cloud / remote chain |
| TTS | **CosyVoice2-0.5B** local streaming server (female zero-shot), on **vLLM in WSL** (TTFB ~1.1s) | `:8001`, repo `E:\Claude\cosyvoice-local-tts` |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait) | `:8002`, `musetalk` conda env |

WebRTC → browser at `http://localhost:7860/client/`. Goal: time-to-first-output **< 8 s**.

TTS / ElevenLabs / Deepgram-Aura are deliberate **fallback switches** via `TTS_PROVIDER`, not
multi-provider branching.

## ⭐ Weather-bot mode + local memory harness (2026-06-23, new in this baseline)

The LLM node is now a **deliberate switch** (`LLM_PROVIDER`, like `TTS_PROVIDER`): `openrouter`
(default, general chat) or **`weather_chain`** — a dedicated Chinese weather bot backed by a remote
NCU LangServe chain (`POST .../chain/resWeatherChain/stream`, `{"input":{"query","model"}}`). The
chain is **stateless + Chinese-only**, so the virtual human's **growing memory lives in a fully-local
harness wrapped around it** (`AVATAR_MEMORY=1`): a `MemoryStore` (profile + rolling summary + session
log under `state/avatar_memory/`) that **rewrites** the utterance into a self-contained zh query via
local **`qwen2.5:3b-cpu`** (Ollama, CPU-pinned so the GPU stays free; ~0.77s/rewrite) before the
chain call, and **distills** the conversation into durable memory on disconnect *and* recovers
leftover turns on startup. To run it: `LLM_PROVIDER=weather_chain LANGUAGE=zh`. NCU was down during
build, so `scripts/mock_weather_chain.py` (:8077) is a local stand-in (same path + LangServe SSE,
answers via CPU qwen) — verified live end-to-end (TTFO 5.7s). Full detail: the
`project-visualllm-weather-chain-memory` memory; key files `local_services/weather_chain_llm.py`,
`local_services/avatar_memory.py`, `pipeline/stages/llm.py`. Tooling: `scripts/probe_weather_chain.py`,
`tools/chat-cpu.html` (a standalone CPU-model chat tester).

## ⭐ Current baseline (2026-06-23, commit `cd88f20`)

Latest work — three things landed/changed today (full write-ups: `docs/PROBLEMS-AND-FIXES.md`
P9/P10 + the run notes below):

1. **Avatar finished ~1–2s BEFORE the audio on long replies — FIXED (P9).** The render server sized
   each segment as `int(16000/fps)*SEG_FRAMES`, but the renderer counts frames as
   `floor(len/sr*fps)`. At an fps that doesn't divide 16000 (e.g. **12**: `int(16000/12)`→1333), an
   8-frame batch rendered **7** → a ~12.5% frame deficit accumulating over the turn, so the lips ran
   short. Fix = `MuseTalkEngine.samples_for_frames(n)=ceil(n*16000/fps)` (server `app.py`), wired into
   stream init / `config` / `_warmup` / the `speech_end` tail-pad. Frame count is now `audio_sec*fps`
   end-to-end. **Keep `MUSETALK_FPS` a divisor of 16000** (8/10/16/20/25…); the fix makes 12 correct
   anyway. Verified: warmup `7→8 frames/segment`; a 13.56s reply renders **163** frames (was 141).
2. **Leftover-audio blip ~1–2s AFTER the turn — KNOWN ISSUE, fix REVERTED by preference (P10).** Once
   the frame count was correct, a fraction-of-a-second of audio still played ~1–2s late (steady's
   `_advance` floor-cap stranded the final sub-frame of audio to the delayed `video_end` drain). A
   `ceil`-the-cap fix was implemented + verified, then **rolled back — the prior behavior was judged
   better**, so the baseline keeps the blip. Root cause + the one-line re-apply path are in
   `docs/PROBLEMS-AND-FIXES.md` P10.
3. **CosyVoice wouldn't start on the shared GPU — FIXED (config).** vLLM's `gpu_memory_utilization`
   was hardcoded `0.2` (a 3.26GB ceiling, below vLLM's own ~4GB footprint → KV cache negative → crash:
   "No available memory for the cache blocks"). Now **env-overridable, default `0.3`**
   (`COSYVOICE_VLLM_GPU_UTIL`, in the **separate `E:\Claude\cosyvoice-local-tts` repo**) → +0.86GB KV /
   75k tokens, fits the ~5.4GB free alongside MuseTalk (it grabs ~2.8GB, leaving ~2.5GB). The 3 GPU
   services (MuseTalk + CosyVoice-vLLM + the desktop) share one 16GB card — if either errors after
   opening another heavy GPU app, that's the VRAM ceiling: close something or nudge the util fraction.

## ⭐ Cleanup (2026-06-22): collapsed to one stack — MuseTalk-only

The codebase was trimmed to a single pure pipeline. **Removed:** the entire Ditto avatar stack
(full-face TensorRT path, its server/client/scripts/plugin build), the `AVATAR=none` audio-only
mode, character/emotion mode (`emotion_tagger`, `CHARACTER_MODE`, the in-character Thai persona),
the debug dashboard (`pipeline/debug/`, `:7861`), the now-fixed-bug audio-garble capture probes
(the CosyVoice noise detector, the transport/handle-audio probes, the MuseTalk downstream capture),
and one-off/superseded probe scripts. The steady-screech regression test + the sync-routing test
were **moved to `archive/`** (kept, not deleted). Stale planning docs and superseded workflow
visualizations were removed. Everything is recoverable from git history.

The avatar is now **MuseTalk** with no engine switch; `AVATAR_REF` sets the portrait.

## ⭐ Avatar lag root-caused + fixed (the long debug session)

Two real avatar-timing bugs and the steady-mode screech, all fixed and measured:

1. **The avatar started ~2s late / "audio ends then the avatar keeps moving" on long replies = a
   per-turn cuDNN re-autotune spike.** `musetalk_server/app.py` had `cudnn.benchmark = True`, but
   the turn-START segment has a different shape than mid-turn, so cuDNN re-ran its expensive autotune
   on the **first segment of every turn** → a ~16s GPU spike. **Fix: `cudnn.benchmark = False`.**
   First-segment GPU 16,372ms → 346ms; lips-start +5.3s → +1.0s; server warmup 17.7s → 1.0s.
   **This is load-bearing — keep it `False`.** See PROBLEMS-AND-FIXES P1.

2. **Residual ~1.9s lip-start lag = the renderer was starved at turn start.** The real-time-paced
   feed couldn't prime its lead frames until audio trickled in. **Fix: burst the first
   `MUSETALK_FEED_BURST_S`=1.0s of a turn's audio un-paced, then resume pacing.** Lip-start
   ~1.9s → ~0.75–1.0s.

3. **`MUSETALK_SYNC_MODE=steady` intermittently SCREECHED the voice; now FIXED + the default.**
   Traced at the byte level: the screech is a **1-byte (odd) sample misalignment, not generated
   noise**. pipecat's output transport fires `_bot_stopped_speaking()` when no audio reaches its
   queue for `BOT_VAD_STOP_FALLBACK_SECS` (3s) and that handler discards the partial `_audio_buffer`.
   In steady the voice is released paced to rendered video, so a >3s render stall starves the queue →
   the 3s timeout fires mid-turn → the odd partial buffer is discarded → the rest of the turn is
   odd-misaligned = screech. **Two-layer fix:** `main.py::_relax_bot_vad_stop_timeout()` raises the
   timeout (we already drive an explicit `TTSStoppedFrame` per turn, so the gap fallback is
   redundant), AND `musetalk_video.py::_align_even` carries any dangling odd byte between downstream
   frames so the PCM stays whole-sample (any buffer clear can then only drop an even gap). Verified by
   `archive/_screech_repro_test.py` + byte-identical pre/post-transport audio. See PROBLEMS-AND-FIXES P3.

## ⭐ CosyVoice on vLLM (TTFB 3.4s → ~1.1s) — the real lip-lag fix

The avatar lip-lag root cause was **CosyVoice's first-chunk latency** (the autoregressive LLM
prefill+gen), not the avatar render. Lead-frames/burst-feed/voice-align only move or freeze it.
The fix = shrink the gap: **CosyVoice2's LLM moved onto vLLM, in WSL Ubuntu on the Blackwell 5060
Ti** → measured TTFB 3.4s → ~1.1s, and it now actually streams. The pipeline reaches it at the
**WSL IP** (NOT localhost — WSL2's localhost relay buffers the audio ~2s). Run with
`bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh`; revert to the Windows PyTorch server via
`COSYVOICE_URL=http://localhost:8001`. Full build notes + gotchas: the
`project-visualllm-cosyvoice-vllm` memory.

## ⭐ Reliability fixes (remote)

- **Intermittent remote mic = WebRTC ICE candidate pollution.** The box advertises many host
  candidates (Tailscale, Hyper-V, Radmin, LAN, APIPA); ICE could nominate a dead pair and drop the
  audio track mid-call. **Fix:** `main.py::_restrict_ice_to_subnet()` keeps only `WEBRTC_ICE_SUBNET`
  (default `100.64.0.0/10`, Tailscale's range). `0` disables.
- **Remote avatar lag = oversized WebRTC stream, not the network/render.** Fit the stream to the
  link: a smaller frame (`MUSETALK_SIZE`) + a bounded VP8 ceiling (`WEBRTC_VIDEO_BITRATE_MAX`, capped
  in `main.py::_configure_webrtc_video_bitrate()`), then a modest receive-side jitter buffer
  (`CLIENT_JITTER_BUFFER_MS`, injected by `_install_client_jitter_buffer()`). Counter-intuitively,
  bitrate that is too LOW starves the VP8 encoder → choppier — don't shrink-and-starve. Isolate
  link-vs-render with `scripts/stream_live.py`.
- **DEBUG log flood** that choked the realtime loop: `log_setup.py` pins the stdlib root to INFO and
  aiortc/aioice to WARNING.

## A/V sync — the architecture decision (read before touching sync)

The HARD constraint: **MuseTalk and CosyVoice share ONE GPU.** MuseTalk renders ~20 fps alone but
CosyVoice bursts the GPU while streaming a reply and slows the render unpredictably. Two sync modes:

- **steady (DEFAULT, `MUSETALK_SYNC_MODE=steady`) — VIDEO-MASTER:** the voice is held and released
  locked to rendered frames for a **synced start** (the user's pick). Per-frame pinning via
  `sync_with_audio` (transport non-live). The old screech is fixed (above). Remaining tradeoff: under
  a long render stall the voice briefly **pauses** then resumes clean.
- **live — AUDIO-MASTER:** the voice is forwarded immediately so it **can never freeze**; the lips
  are best-effort (~0.75s trail under contention). The robust alternative if the steady pause is
  worse than the lip trail.

**Do NOT re-lock the voice to video as a global default** — fully locked/video-master sync froze the
voice on any render stall (confirmed). If the lips trail too far in `live`, the SAFE lever is
bounding the avatar server's `out_q` (drop stale frames), never re-locking the voice.

**Critical coupling (`main.py`):** `sync_with_audio` is a no-op unless the transport is non-live in
pipecat 1.3.0, so `video_out_is_live = not config.avatar_sync_with_audio`. One fps everywhere
(server stride, client clock, `video_out_framerate`) = `MUSETALK_FPS` or A/V drifts.

## Known tradeoffs (accepted)

- **Echo-guard on by default** → use headphones (or `ECHO_GUARD=0` to allow barge-in, relying on OS
  echo cancellation).
- **Single shared GPU** → under heavy contention the lips can trail the voice in `live` mode. A
  genuine fix needs a dedicated avatar GPU or a TensorRT'd MuseTalk (fp16 is on; no TRT yet).
- **conda env cert store** is broken in `musetalk`/`tts` → curl-cache weights + set `SSL_CERT_FILE`
  (certifi). See the `project-visualllm-conda-ssl-weights` memory.

## Key files

- `pipeline/main.py` — pipeline assembly, LLM warmup, greeting, the screech fix + the three
  remote-WebRTC fixes (ICE pin, bitrate cap, jitter-buffer inject).
- `pipeline/config.py` — keys + `LANGUAGE` (en/zh/th) + the MuseTalk avatar knobs, driven by `.env`.
- `pipeline/stages/*.py` — per-stage factories (stt/llm/tts/avatar/vad).
- `pipeline/metrics.py` — `TtfoMeter` (logs `[TTFO]` per turn + summary).
- `local_services/musetalk_server/app.py` — MuseTalk GPU server (ws, frame-clock `pump()` markers,
  single-client session guard, watchdog).
- `local_services/musetalk_video.py` — Pipecat client for the MuseTalk server; owns the
  frame-clocked A/V sync (`_feed_q` pacing, `_align_even` anti-screech guard).
- `local_services/cosyvoice_tts.py` — CosyVoice streaming TTS client.
- `scripts/preflight.py` — import/drift check. `scripts/measure.py` — unified A/V timing harness.
- `archive/` — kept-out-of-tree regression tests.

## Open / next

- **Live lip-sync tuning (human check):** open `/client/` with a real mic + headphones, speak a
  multi-sentence turn, and judge steady vs live. Tune `MUSETALK_SYNC_LEAD_S` if the lips lead/trail.
- Confirm the felt CosyVoice-vLLM latency improvement on a live remote call.
