# VisualLLm — Project Status & Next Steps

_Last updated: 2026-06-30 (**Added local OFFLINE STT** — `STT_PROVIDER=sherpa` (sherpa-onnx streaming, in-process, CPU/~0 VRAM, recommended) + `funasr` (SenseVoice segmented, `:8004`); default stays Deepgram. On branch `feat/offline-stt-sensevoice`, not pushed. See the session note below. Earlier: **LLM reverted to cloud (gemini-2.5-flash-lite)** per the 2026-06-29 plan — fixes the CPU-contention regression. Investigated a **Chinese-only voice-delay**: root cause is CosyVoice's zh first-chunk TTFB (~2.3s vs en ~1.1s), NOT the LLM/avatar — but the fix conflicts with the avatar on the shared GPU, so it's left UNRESOLVED, see the 2026-06-30 session below + P15. Previously (2026-06-29): MOSS-TTS option, web CONFIG PANEL, local-Ollama LLM mode.)_

> **See `WORKFLOW.md`** for the full end-to-end system workflow (the processes, the turn
> flow, the avatar wire contract, running locally + remote, config reference).
> **See `docs/PROBLEMS-AND-FIXES.md`** for the catalogue of bugs found + how each was fixed.

## The stack

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en`/`zh`/`th` by `LANGUAGE`) default; local OFFLINE alt `STT_PROVIDER=sherpa` (streaming, in-process) or `funasr` (segmented, `:8004`) | cloud / **local CPU** |
| LLM | `LLM_PROVIDER=openrouter` (OpenAI-compatible — **cloud OR local Ollama** by `OPENROUTER_BASE_URL`) or `weather_chain` (Chinese weather bot) | cloud / local / remote chain |
| TTS | **CosyVoice2-0.5B** local streaming (default), or **MOSS-TTS-Realtime** (`TTS_PROVIDER=moss`) | `:8001` cosy (WSL) / `:8003` moss (WSL) |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait) | `:8002`, `musetalk` conda env |
| Config | **Web config panel** — edit `.env` + restart pipeline from the browser | `:7870` (`:8444` over Tailscale) |

WebRTC → browser at `http://localhost:7860/client/`. Goal: time-to-first-output **< 8 s**.

TTS providers (`cosyvoice` default · `moss` · `elevenlabs` · `deepgram`) and LLM providers
(`openrouter` · `weather_chain`) are deliberate **single-provider switches** via `.env`, not
multi-provider branching. **Easiest way to change any of this: the config panel (`:7870`).**

## ⭐ Session 2026-06-30 (later): local OFFLINE STT added (sherpa streaming + funasr segmented)

Branch `feat/offline-stt-sensevoice` (off `main`, **not pushed**). Goal: replace cloud Deepgram with a
fully-local STT. Two options added behind `STT_PROVIDER` (default stays `deepgram`):

- **`sherpa` (recommended)** — sherpa-onnx streaming zipformer, **bilingual zh-en**, **in-process** (system
  Python, no server), **CPU/~0 VRAM**. zh→Traditional via OpenCC. `local_services/sherpa_stt.py`. Model under
  `models/` (gitignored). Knob `SHERPA_ENDPOINT_SILENCE` (0.5s default) = pause that fires the query.
- **`funasr`** — SenseVoice-Small segmented server (`:8004`, `funasr-stt` conda env). `local_services/funasr_server/`.

**Root cause that drove the design (live-debugged, don't re-derive):** on this box's **remote/RDP mic**, audio
reaches the STT (~700 frames) but Pipecat's **energy-VAD never fires `VADUserStoppedSpeakingFrame`**, so the
`LLMUserAggregator` never flushes the user turn → no LLM response. Deepgram survived only because it's
streaming + self-endpointing; **segmented SenseVoice depends on that VAD, so it was fatal.** Fix: **sherpa
streaming drives turn-taking from its own ASR endpoint** (emits the VAD start/stop frames itself), so the turn
flushes regardless of the energy-VAD. **Verified end-to-end via the WebRTC probe** (synthetic mic → sherpa →
LLM answered a weather question → CosyVoice spoke it). **Still OPEN:** the user's **real-mic** confirmation,
and `[TTFO]` meter reads `count:0` with ASR-driven turns (turn works; meter wiring to revisit). Box has **no
physical mic** (RDP "Remote Audio" only) — needs RDP mic redirection for an on-box test. Specs/plan:
`docs/superpowers/{specs,plans}/2026-06-30-local-offline-stt*`. Full reference: `INSTALL.md` §6.5, `WORKFLOW.md`.

## ⭐ Session 2026-06-30: LLM reverted to cloud; Chinese voice-delay diagnosed (shared-GPU conflict)

**1. Reverted the LLM to cloud (the 2026-06-29 plan).** `.env` now: `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`,
`OPENROUTER_MODEL=google/gemini-2.5-flash-lite` (+ a real `OPENROUTER_API_KEY`). This frees the CPU (the
local CPU-pinned `qwen2.5:3b-cpu` was the contention source) and gives clean, coherent Chinese text instead
of the small-model fragments (`你好！…继续吗？`). Keep this.

**2. Diagnosed the "Chinese voice starts later than English" complaint → it is the TTS, and it CONFLICTS with the avatar (P15, NOT RESOLVED).**
Measured at the boundary: CosyVoice first-audio TTFB is **~2.3 s for Chinese vs ~1.1 s for English** because
CosyVoice emits a **larger opening stream chunk for zh (~4.4 s of audio vs ~2 s)** before yielding. Ruled out
(by experiment): text-normalization (no `wetext`/`ttsfrd` frontend in the WSL env), the zero_shot-vs-cross_lingual
path (forcing zh through cross_lingual didn't change TTFB), and the LLM. The existing `COSYVOICE_FIRST_HOP=5`
knob cuts zh TTFB to ~1.25 s **but** its extra small GPU inferences starve MuseTalk's render on the shared 16 GB
card (avatar lips-start jumped ~2 s → ~8 s) — so it was **reverted**. **On one GPU, fast zh TTS and a smooth
avatar are mutually exclusive; the real fix is a dedicated avatar GPU, not a setting.** Full detail: P15.

**3. Shared-GPU restart order (learned the hard way).** CosyVoice's vLLM must load **before** MuseTalk or it
crashes `No available memory for the cache blocks`. Recovery / clean baseline = stop all → start cosyvoice on
the near-empty card → then `scripts/run.ps1` (MuseTalk + pipeline). Healthy VRAM with all three ≈ 300–400 MB free.

---

## ⭐ Session 2026-06-29: MOSS TTS option, web config panel, local-LLM mode, real NCU found

> **⚠️ HONEST CAVEAT (end of session).** The features below were ADDED and work in isolation, but the
> live experience **regressed**: the MOSS between-sentence delay was **not** resolved and overall latency
> got **worse** (P13). Leading hypothesis: **CPU contention** — this session ran the LLM on a CPU-pinned
> local Ollama plus the memory harness / memory-sim / weather mock all on CPU, while the GPU ran
> CosyVoice-vLLM + MuseTalk; the original smooth baseline used a **cloud** LLM, leaving the CPU free.
> **Plan: next session revert `.env` to the baseline (cloud LLM + CosyVoice + "weather" voice + steady)
> and re-measure TTFO before re-trying any of this.** The new CODE is inert until `.env` selects it, so
> the revert is purely an `.env` change. Baseline values: `WORKFLOW.md §8` + `CLAUDE.md`.

Four things landed this session (all `.env`-switchable; nothing removed):

1. **MOSS-TTS-Realtime as a second TTS provider (`TTS_PROVIDER=moss`).** A streaming server
   (`local_services/moss_server/app.py`, `moss-tts` conda env, `:8003`) that speaks the **same
   `/tts/stream` raw-PCM wire contract as the CosyVoice server**, so the pipeline reaches it through
   the existing CosyVoice client just by repointing `MOSS_URL`. Voice = a fixed reference clip
   (`MOSS_REF`, clone-only). **Streaming is load-bearing:** the first cut synthesized the whole
   sentence before sending audio (TTFB ≈ full gen ≈ 8.5s); the streaming rewrite (MOSS's
   `MossTTSRealtimeStreamingSession` + `AudioStreamDecoder`) drops TTFB to **~0.4s warm**. Run it
   **eager** (`TORCHDYNAMO_DISABLE=1`, the server's default) — compiled mode recompiles ~3–40s on each
   new sentence-length and that spikiness reads as "delay between sentences". Needs `CC`/`CXX` (triton)
   + the `torchcodec` ffmpeg-7 + `nvidia-npp` + `LD_LIBRARY_PATH` fix; launch recipe is in the server
   docstring. The no-recompile production path is vLLM-Omni (next step). CosyVoice remains the default.

2. **Web config panel (`local_services/config_panel/`, `:7870`).** A stdlib server + single-page UI to
   **view/edit `.env` and restart the pipeline from the browser** (incl. remotely over Tailscale at
   `:8444`). Curated dropdowns (language, LLM/TTS provider + model/voice, sync mode, memory) + an
   advanced section (URLs, FPS, lead frames, jitter, ICE…), live server-status dots, Save (writes
   `.env` **in place, preserving comments**), and Restart. **Restart kills `:7860` via a native Win32
   `TerminateProcess`, NOT `taskkill`/PowerShell** — those hang for tens of seconds on this box under
   CPU load (the bug that first made Restart error out).

3. **LLM can run fully local.** The `openrouter` branch is just an OpenAI-compatible client, so pointing
   `OPENROUTER_BASE_URL=http://localhost:11434/v1` + `OPENROUTER_MODEL=qwen2.5:3b-cpu` runs a
   **CPU-pinned local Ollama** model as the chat LLM (measured **TTFB ~0.5s**, good zh, no GPU). The
   CPU pin matters — the GPU is full of CosyVoice-vLLM + MuseTalk (~680MB free). Swap to `qwen2.5:7b`
   for better quality (slower on CPU).

4. **Real NCU weather chain found again + a professional CosyVoice voice.** NCU moved off `:8000` to
   **HTTPS/443** with a self-signed cert (`WEATHER_CHAIN_URL=https://140.115.54.87/chain/resWeatherChain`,
   `WEATHER_CHAIN_VERIFY_TLS=0`). Verified live: only **`qwen2.5:7b`** (~1.2s) and `gemma3:27b` (~3.85s)
   are installed; the lighter qwen/gemma sizes 500. CosyVoice's reference voice is now **env-driven**
   (`COSYVOICE_PROMPT_WAV` / `COSYVOICE_PROMPT_TEXT` / `COSYVOICE_SPK_ID` in `cosyvoice-local-tts/tts_engine.py`,
   defaulting to the original "weather" speaker) so a professional voice is a config choice, not a hardcode.

New tooling: **`scripts/_capture_synced.py`** — like `_capture.py` but keeps ONLY the real frames between
the server's `video_start`/`video_end` markers (auto-detecting frame size), so the offline mp4 is A/V-synced
(the old probe kept the neutral lead frames → lips trailed the muxed audio). Research: a full
**Chinese-TTS-alternatives comparison** under `research/chinese-tts-alternatives/` (REPORT.md + per-model JSON).

## ⭐ Weather-bot mode + local memory harness (2026-06-23, new in this baseline)

The LLM node is now a **deliberate switch** (`LLM_PROVIDER`, like `TTS_PROVIDER`): `openrouter`
(default, general chat) or **`weather_chain`** — a dedicated Chinese weather bot backed by a remote
NCU LangServe chain (`POST .../chain/resWeatherChain/stream`, `{"input":{"query","model"}}`). The
chain is **stateless + Chinese-only**, so the virtual human's **growing memory lives in a fully-local
harness wrapped around it** (`AVATAR_MEMORY=1`): a `MemoryStore` (profile + rolling summary + session
log under `state/avatar_memory/`) that **rewrites** the utterance into a self-contained zh query via
local **`qwen2.5:3b-cpu`** (Ollama, CPU-pinned so the GPU stays free; ~0.77s/rewrite) before the
chain call, and **distills** the conversation into durable memory on disconnect *and* recovers
leftover turns on startup. To run it: `LLM_PROVIDER=weather_chain LANGUAGE=zh`. **NCU is back up
(2026-06-29) at `https://140.115.54.87/chain/resWeatherChain` — HTTPS/443, self-signed cert, so set
`WEATHER_CHAIN_VERIFY_TLS=0`; `qwen2.5:7b` is the fast working model (~1.2s).** `scripts/mock_weather_chain.py`
(:8077) remains a local stand-in (same path + LangServe SSE, answers via CPU qwen) for when NCU is down.
Full detail: the
`project-visualllm-weather-chain-memory` memory; key files `local_services/weather_chain_llm.py`,
`local_services/avatar_memory.py`, `pipeline/stages/llm.py`. Tooling: `scripts/probe_weather_chain.py`,
`tools/chat-cpu.html` (a standalone CPU-model chat tester).

**Chain model latency (measured 2026-06-24, NCU live).** The whole weather turn is dominated by the
**remote chain's generation time** (STT + local memory-rewrite + CosyVoice ~2.5s are minor; MuseTalk
is off the critical path). `WEATHER_CHAIN_MODEL` picks the model NCU runs. Timed against the live NCU
server with the same zh question:

| `WEATHER_CHAIN_MODEL` | time-to-answer | notes |
|---|---|---|
| `gemma3:27b` (old default) | **~21–33 s** | cleaner Traditional zh, but ~20× slower |
| **`qwen2.5:7b`** (now the `.env` default) | **~1.5 s warm / ~14 s cold** | answer grounded + correct; occasionally mixes a simplified char / "Taipei" |
| `gemma3:12b/4b`, `qwen2.5:3b`, `llama3.1:8b` | n/a | **not pulled on NCU** (chain closes the stream → `RemoteProtocolError`) |

So `qwen2.5:7b` is the fastest model NCU actually serves — the practical floor. The **first** query of a
session can still be ~10–15 s while NCU warms/loads the model, then it's ~1.5 s. (config.py default is
still `gemma3:27b`; the live `.env` overrides to `qwen2.5:7b`.)

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
2. **Leftover-audio blip ~1–2s AFTER the turn — FIXED (P10, re-applied 2026-06-24).** Once the frame
   count was correct, a fraction-of-a-second of audio still played ~1–2s late (steady's `_advance`
   floor-cap `int()` stranded the final sub-frame of audio to the delayed `video_end` drain). Fix =
   `int()`→`math.ceil` on the `audio_cap` (`musetalk_video.py::_advance`) so the trailing frame is
   reachable and the last sub-frame releases in step. (Historically reverted-by-preference; verified
   2→0 stranded chunks and re-applied this session.) `docs/PROBLEMS-AND-FIXES.md` P10.
3. **CosyVoice wouldn't start on the shared GPU — FIXED (config).** vLLM's `gpu_memory_utilization`
   was hardcoded `0.2` (a 3.26GB ceiling, below vLLM's own ~4GB footprint → KV cache negative → crash:
   "No available memory for the cache blocks"). Now **env-overridable, default `0.3`**
   (`COSYVOICE_VLLM_GPU_UTIL`, in the **separate `E:\Claude\cosyvoice-local-tts` repo**) → +0.86GB KV /
   75k tokens, fits the ~5.4GB free alongside MuseTalk (it grabs ~2.8GB, leaving ~2.5GB). The 3 GPU
   services (MuseTalk + CosyVoice-vLLM + the desktop) share one 16GB card — if either errors after
   opening another heavy GPU app, that's the VRAM ceiling: close something or nudge the util fraction.

## ⭐ Session 2026-06-24: breathing idle removed; smooth mouth-close investigated + ABANDONED

1. **Breathing idle removed (user pick).** `MUSETALK_IDLE_MOTION=0` — between turns the face holds the
   static neutral portrait instead of the synthesized breathing/sway loop. Server reads it from OS env,
   so `run.ps1` now propagates it.
2. **Smooth end-of-turn mouth-close — INVESTIGATED, NOT SHIPPED (kept the clean snap).** _[SUPERSEDED
   2026-06-27 — now FIXED via the free-run close crossfade; see the 2026-06-27 session below.]_ The mouth
   snaps to the resting face at end of turn. Root cause is a **rendered→photo domain jump** (every
   MuseTalk frame sits ~5px from the neutral *photo* because rendered frames come from the VAE), not an
   open/closed-mouth jump — so a silence-rendered tail did NOT help. A pixel **crossfade** (last spoken
   frame → neutral) is smooth at the SERVER, but in **steady mode it cannot be delivered**: the non-live
   transport paces video by the interleaved real-time **audio** frames, and the close has no audio, so
   trailing frames can't be clocked — three delivery attempts (burst / `asyncio.sleep` pacing /
   silent-audio pairing) each collapsed or jittered (proven at the WebRTC delivery path). **Conclusion:
   a smooth close needs either `live` mode (video free-runs on its own clock) — but the user rejected
   live for the ~0.75s lip trail — or a rendered-rest-pose (hold a VAE-rendered closed-mouth frame as
   the rest pose so the domain pop ~vanishes; untried). All close-crossfade code was reverted; the end
   is the clean snap.** Full write-up: `docs/PROBLEMS-AND-FIXES.md` P12. **KEPT:** the P10 ceil
   audio-blip fix (above).

## ⭐ Session 2026-06-27: choppy close FIXED — free-run close crossfade (steady keeps synced lips)

The end-of-turn mouth snap is fixed. At `video_end` the client cross-dissolves the last spoken frame ->
the rest pose over `MUSETALK_CLOSE_FADE_FRAMES` (5 = ~0.42s) frames, delivered **free-running** (untagged,
paced at fps, like the idle loop) so video runs on its own clock just for the close — **"live during the
close", steady through the speech.** `musetalk_video.py::_play_close_fade`. Supports: `MUSETALK_END_TAIL_FRAMES=0`
(so the last buffered frame is the last SPOKEN frame) + a server pump change so it settles to the NEUTRAL
rest pose when idle even with END_TAIL=0 (`musetalk_server/app.py`). Two earlier tries that DIDN'T work,
both ruled out on the delivery path: (1) rest-pose swap (only halves the *domain* pop, mouth still
shape-snaps), reverted; (2) audio-PAIRED crossfade (the `_advance` audio-cap strands the close frames
when the render runs behind). Free-run sidesteps both. **Verified on the WebRTC delivery path** (not the
offline capture, which bypasses the transport): the mouth-to-rest distance now ramps over ~5 frames
(snap-index 0.92 -> 0.58) instead of one big step. Set `MUSETALK_CLOSE_FADE_FRAMES=0` (+ END_TAIL>0) for
the old clean snap. `docs/PROBLEMS-AND-FIXES.md` P12. Delivery-path tooling: `scratchpad/probe_close.py`.

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

- **Echo-guard now defaults OFF** (`ECHO_GUARD=0`, barge-in) → **use headphones** (or OS echo
  cancellation) so the live mic doesn't pick up the avatar. The half-duplex mute (`=1`) is broken
  under the default `steady` sync mode — it leaves the mic stuck-muted after a turn so voice never
  triggers; see `docs/PROBLEMS-AND-FIXES.md` P11. Only use `=1` with `MUSETALK_SYNC_MODE=live`.
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
- `local_services/cosyvoice_tts.py` — CosyVoice streaming TTS client (also reused by `TTS_PROVIDER=moss`).
- `local_services/moss_server/app.py` — MOSS-TTS-Realtime streaming server (same wire contract as cosy).
- `local_services/config_panel/` — the web config panel (`server.py` + `index.html`, `:7870`).
- `scripts/preflight.py` — import/drift check. `scripts/measure.py` — unified A/V timing harness.
  `scripts/_capture_synced.py` — A/V-synced offline avatar capture (real frames only).
- `archive/` — kept-out-of-tree regression tests.

## Open / next

- **Live lip-sync tuning (human check):** open `/client/` with a real mic + headphones, speak a
  multi-sentence turn, and judge steady vs live. Tune `MUSETALK_SYNC_LEAD_S` if the lips lead/trail.
- Confirm the felt CosyVoice-vLLM latency improvement on a live remote call.
