# VisualLLm — Project Status & Next Steps

_Last updated: 2026-06-22 (evening)_

> **See `WORKFLOW.md`** for the full end-to-end system workflow (the two processes, the
> turn flow, the avatar wire contract, running locally + remote, config reference).
> **See `docs/PROBLEMS-AND-FIXES.md`** for the full catalogue of bugs found + how each was
> fixed (this session's deep debug included).

## ⭐⭐⭐ Session 2026-06-22 (evening): avatar lag root-caused + fixed; steady-mode screech traced to pipecat

A long systematic-debugging session on "the avatar lags / the voice screams". Net: **two real
avatar-timing bugs fixed and measured**, and the **steady-mode voice screech reliably root-caused
to a pipecat library bug** (not our code) — so the system stays on **`MUSETALK_SYNC_MODE=live`**.
Full per-problem writeup: **`docs/PROBLEMS-AND-FIXES.md`**.

**1. ✅ FIXED — the avatar started ~2s late / "audio ends then the avatar moves for ~10s" on long
replies = a per-turn cuDNN re-autotune spike.** `musetalk_server/app.py` set
`torch.backends.cudnn.benchmark = True` ("fixed shapes"), but the **turn-START segment has a
different shape** than mid-turn segments, so cuDNN re-ran its expensive algorithm autotune on the
**first segment of every turn** → a **~16 s GPU spike** on this shared card (profiled:
`gpu=16372ms` first segment, `240ms` after). That made the first lip frame land ~5 s late and the
render fall hopelessly behind on long replies. **Fix: `cudnn.benchmark = False`.** Measured:
first-segment GPU **16,372 ms → 346 ms**, lips-start **+5.3 s → +1.0 s**, a 13 s reply now renders
179 frames at a steady 12 fps. Bonus: server warmup **17.7 s → 1.0 s**.

**2. ✅ FIXED — residual ~1.9 s lip-start lag = the renderer was starved at turn start.** The
real-time-paced feed (`musetalk_video.py _feed_q`) also starved the renderer at the *start* of a
turn (it couldn't fill its lead-frame prime until audio trickled in). **Fix: burst the first
`MUSETALK_FEED_BURST_S`=1.0 s of a turn's audio un-paced, then resume pacing.** Measured lip-start
**~1.9 s → ~0.75–1.0 s**, no backlog. (Also added a one-shot **`_warmup()`** at server load and a
per-turn **`[avatar timing]`** log: `lips start +Xs after voice | … | end drift +Ys`.)

**3. ✅ FIXED (2026-06-22 night) — `MUSETALK_SYNC_MODE=steady` intermittently SCREECHED the voice
mid-reply; steady is now the default.** The earlier conclusion ("unfixable pipecat non-live
write-path bug, stay on live") was **wrong** — that boundary table came from non-time-aligned runs +
per-frame RMS false positives. Re-traced at the **byte level**: the screech is a **1-byte (odd)
sample misalignment, not generated noise** (shifting the garbled region by 1 byte recovers clean
speech: flatness 0.52→0.079). Byte-diffing the clean MuseTalk-push stream vs the delivered stream:
identical until 6.040s, then **1049 bytes (odd) DELETED**, rest bit-identical but odd-shifted →
broadband noise to turn end. **Root cause:** pipecat's output transport fires `_bot_stopped_speaking()`
when no audio reaches its queue for `BOT_VAD_STOP_FALLBACK_SECS` (**3s**), and that handler does
`self._audio_buffer = bytearray()` — discarding the partial buffer. In steady the voice is released
**paced to rendered video**, so a **>3s render stall** starves the queue → the 3s timeout fires
**mid-turn** → the odd partial buffer is discarded → the rest of the turn is odd-misaligned = screech.
(Live forwards audio continuously, never a 3s gap — that's why it was clean.) **Fix:**
`main.py::_relax_bot_vad_stop_timeout()` raises that timeout (knob `BOT_VAD_STOP_FALLBACK_SECS`,
default 600s); we already drive an explicit `TTSStoppedFrame` per turn so the audio-gap fallback is
redundant. A stall now just pauses the voice and it resumes clean. **Verified:** deterministic
`scripts/_screech_repro_test.py` (drives the real `MediaSender`: 3s timeout discards the partial
buffer = bug; raised timeout keeps it = fix) **PASS**; live steady turns deliver **byte-identical**
pre/post-transport audio (0 noisy windows). Full writeup: `docs/PROBLEMS-AND-FIXES.md` P3.

**New tooling this session:** `scripts/measure.py` (one-command turn timeline → `output/
measure_report.json` + `docs/measure_data.js`, auto-rendered by `docs/workflow-timeline.html`),
`docs/gpu-timeline.html` (CosyVoice-vs-MuseTalk GPU share), and env-gated audio-debug captures
(all default OFF): `COSYVOICE_DELIVERED_CAPTURE`, `COSYVOICE_HANDLE_AUDIO_PROBE`,
`COSYVOICE_YIELD_PROBE`, `MUSETALK_DOWNSTREAM_CAPTURE`, `COSYVOICE_NOISE_LOG`/`CAPTURE_ALL`.

**Current accepted `.env`:** `MUSETALK_SYNC_MODE=live`, `MUSETALK_LEAD_FRAMES=14` (load-bearing —
lower starves → freeze), `MUSETALK_END_TAIL_FRAMES=10` (softer mouth-close, user's choice),
`MUSETALK_FEED_BURST_S=1.0`, plus the `cudnn.benchmark=False` code fix.

---

## ⭐⭐ Session 2026-06-22: reliability fixes + CosyVoice on vLLM (TTFB 3.4s → ~1.1s)

A "works sometimes, mostly not / not smooth" report was traced to **three** separate causes,
all fixed, plus the TTS latency root cause was finally killed by moving CosyVoice to vLLM.

**1. Remote mic died most turns = WebRTC ICE candidate pollution (THE intermittency).**
The user views over a remote Tailscale browser; every session ended `TTFO {'count': 0}` with zero
transcripts + `Media stream error; clearing track`. NOT bandwidth/TURN — the box advertised **9 host
candidates** (Tailscale, Hyper-V, Radmin, LAN, APIPA); ICE checked a dead matrix, nominated a marginal
pair, then `Consent to send expired` dropped the audio track. The Tailscale pair is verified reachable.
**Fix:** `main.py::_restrict_ice_to_subnet()` monkeypatches `aioice.get_host_addresses` to keep only
`WEBRTC_ICE_SUBNET` (default `100.64.0.0/10`, Tailscale's range) → 9 candidates collapse to 1 working one.

**2. DEBUG log flood choking the realtime loop.** `log_setup.py` defaulted to DEBUG with the stdlib root
at level 0 → aiortc logged every RTP packet through a stack-walking handler ON the asyncio media loop
(41k lines / 10MB per 20min, and pressure that aggravated #1). **Fix:** intercept root → INFO + aiortc/
aioice → WARNING. Verified 41,492 → 0 DEBUG lines.

**3. Avatar lips trail the voice ~1.5-2s.** Measured the real cause: NOT the avatar render (server
starts lips in 0.77s if fed fast) but **CosyVoice's first-chunk latency** — it delivered the opening
~1.2s of speech over ~1.5-3s, a ~3s FIXED per-reply cost (the autoregressive LLM prefill+gen). Proven
that lead-frames/burst-feed/voice-align all only move or freeze it, never fix it (do NOT retry those —
`MUSETALK_LEAD_FRAMES=14` is load-bearing; lower freezes). The only real fix = shrink the gap:

**→ CosyVoice2 LLM moved onto vLLM, in WSL Ubuntu on the Blackwell 5060 Ti. Measured TTFB 3.4s → ~1.1s**
(and it now actually streams). The server runs in WSL (`cosyvllm` conda env, vllm 0.23 + torch 2.11/
cu130); the Windows pipeline reaches it at the **WSL IP** (NOT localhost — WSL2's localhost relay buffers
the audio stream ~2s). Full run instructions + the dozen bleeding-edge gotchas (model registration vLLM
was missing, `embed_input_ids` rename, flashinfer/Triton JIT → conda compiler, torchcodec, env vars) are
in the `project-visualllm-cosyvoice-vllm` memory and `E:\Claude\cosyvoice-local-tts\run_vllm_server.sh`.
Revert: `.env COSYVOICE_URL=http://localhost:8001` + start the Windows CosyVoice server. Eager mode (no
CUDA graphs) leaves some headroom. **Pending: the user's live-call confirmation of the felt improvement.**

---

## ⭐ Current state (2026-06-21): MuseTalk female avatar + CosyVoice local TTS

The server-render avatar path was reworked to a **fully-local** stack:

| Stage | Service | Port |
|-------|---------|------|
| STT | Deepgram nova-2 | cloud |
| LLM | OpenRouter (`OPENROUTER_MODEL`) | cloud |
| TTS | **CosyVoice2-0.5B** local streaming server (female zero-shot voice) | **:8001** |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait) | **:8002** |

- **Avatar = MuseTalk** (`AVATAR=musetalk`, default), driven by `assets/avatar_female.png` (a
  generated front-facing female portrait via `AVATAR_REF`). Ditto kept as a fallback (`AVATAR=ditto`).
- **TTS = CosyVoice** (`TTS_PROVIDER=cosyvoice`), a SEPARATE repo/process at
  `E:\Claude\cosyvoice-local-tts` (its own `tts` conda env). Added a streaming `/tts/stream`
  endpoint + revived `local_services/cosyvoice_tts.py`. ElevenLabs/Deepgram kept as fallbacks.

**Run it (3 processes):**
```powershell
# 1. CosyVoice TTS server (tts conda env)   -- from E:\Claude\cosyvoice-local-tts
E:\miniconda3\envs\tts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8001
# 2 + 3. MuseTalk avatar server + pipeline (one script, reads AVATAR from .env)
.\scripts\run.ps1
```
Open `http://localhost:7860/client/` (trailing slash). `scripts/run.ps1` starts the right avatar
server by `AVATAR` and propagates `AVATAR_REF / MUSETALK_SIZE / MUSETALK_FPS / MUSETALK_LEAD_FRAMES /
MUSETALK_END_TAIL_FRAMES` to it (the avatar server reads OS env only).

### A/V sync — the hard part (and the architecture decision)
The HARD constraint: **MuseTalk and CosyVoice share ONE GPU.** Profiled, MuseTalk renders ~20 fps
ALONE (keeps up at 12 fps), but CosyVoice **bursts** the GPU while streaming a reply and slows the
render unpredictably. Two sync architectures were tried:

- **steady / locked (REJECTED):** voice released locked to the rendered video → perfect sync when the
  render keeps up, BUT any render stall **freezes the voice too** (total freeze). Confirmed in logs
  (`hold=2.88s audio 2.9s video 0.0s`). **Do not use locked/video-master sync on this shared-GPU setup.**
- **live / audio-master (DEFAULT, `MUSETALK_SYNC_MODE=live`):** the voice is forwarded IMMEDIATELY so it
  **can never freeze**; the lips are best-effort (under contention they briefly hold/trail, but the
  conversation never stops). This is the robust choice.

Supporting fixes that DO help and are kept:
- **Real-time-paced feed** (`_feed_q` in `musetalk_video.py`): the client feeds audio to the renderer at
  real-time so CosyVoice's faster-than-real-time output can't build a video backlog (was the
  "audio finishes but the avatar keeps going 5 s later" lag).
- **CosyVoice GPU pacing** (`COSYVOICE_PACE_RATE=1.3` in the cosyvoice server's `/tts/stream`):
  `time.sleep` after each chunk caps production at 1.3× real-time so the GPU work spreads thin instead
  of bursting → less contention → render keeps up → fewer lip dips. First-chunk latency unaffected.
- **End tail** (`MUSETALK_END_TAIL_FRAMES`) + render the final partial frame so the ending isn't cut.
- Earlier: the odd-length-buffer crash (audio never reached the server → idle-only avatar), the
  server `video_start/clock/end` markers, the single-client **session guard**, and the jitter buffer
  cut 600→150 ms (it was delaying the VOICE; only raise for a real remote/WAN viewer).

### Key `.env` knobs (current)
`AVATAR=musetalk` · `TTS_PROVIDER=cosyvoice` · `MUSETALK_SYNC_MODE=live` · `MUSETALK_FPS=12`
(the rate the render sustains under contention) · `MUSETALK_SIZE=256` (note: shrinking it does NOT
cut MuseTalk compute — fixed internal face resolution) · `MUSETALK_LEAD_FRAMES=4` ·
`COSYVOICE_PACE_RATE=1.3` · `CLIENT_JITTER_BUFFER_MS=150` · `AVATAR_REF=assets/avatar_female.png`.

### Known limitation + next lever
Single shared GPU → under heavy contention the **lips can trail the voice** (live mode, no freeze).
Accepted tradeoff. If it trails too far, the next SAFE step is bounding the avatar server's `out_q`
(drop stale frames so the lips stay current) — **never** re-lock the voice. A genuine fix needs a
dedicated avatar GPU or a TensorRT'd MuseTalk (fp16 is on; no TRT yet). Test tooling:
`scripts/_webrtc_probe.py` (headless aiortc probe; its synthetic-mic→STT path is flaky for multi-turn),
`scripts/_render_offset.py`, `local_services/musetalk_server/_capture.py`. CosyVoice fp16 was tried and
reverted (no gain on the 0.5B model — it's token-gen-bound, not FLOP-bound).

## ⭐ Current focus: the audio-only Thai character (validation sprint)

The active work is **not** the server-rendered talking-head below — it's the **live-visual Thai
character** validation demo for the venture track (`E:\Claude\visualllm-business`). The council said
*test, don't build* (see that repo + the `project-visualllm-validation-sprint` memory), so this
pipeline now runs in a cheaper, thesis-faithful mode:

- **Audio-only mode (`AVATAR=none`)** — the pipeline skips the server GPU face render entirely and
  streams **only the voice** over WebRTC; the **3D avatar is rendered client-side** in
  `visualllm-business/prototype-3d-avatar` (the ~1/50th-cost unit-economics path). No conda/Ditto/CUDA
  needed in this mode.
- **Thai + in-character (`LANGUAGE=th`, `CHARACTER_MODE=1`)** — Deepgram `th`, an in-character Thai
  novel-character system prompt (`น้องเอวา`), and a leading **`[emotion]` tag** per reply that the
  new `stages/emotion_tagger.py` **strips before TTS** (never spoken) and pushes to the client
  (`RTVIServerMessageFrame`) to drive the avatar's face + body-language gesture.

**Run the validation demo (two terminals):**
```powershell
# 1. brain — audio-only Thai character pipeline (no GPU needed)
cd E:\Claude\VisualLLm
$env:AVATAR="none"; $env:LANGUAGE="th"; $env:CHARACTER_MODE="1"; python -m pipeline.main   # :7860
# 2. face — the client-rendered avatar (separate terminal)
cd E:\Claude\visualllm-business\prototype-3d-avatar
npm run dev                                                                                # :5173
```
Open `http://localhost:5173`, click **📞 Live call**, allow the mic, use **headphones** (no
echo-guard), talk in Thai. She lip-syncs, emotes, waves hello, and gestures while she speaks (the
avatar/motion code lives in the `prototype-3d-avatar` README §📞 / §motion).

New `.env` knobs (all optional, default to the old behavior): `AVATAR` (`ditto`|`none`),
`CHARACTER_MODE` (`0`|`1`), and `LANGUAGE` now also accepts `th`.

The **Ditto server-render path below still works** (`AVATAR=ditto`, the default) — it's the
research/photoreal track, kept intact but not the current focus.

---

## 🧹 Cleanup done: one pure stack
The codebase was collapsed from a multi-provider research sprawl to a single
pure pipeline. Removed: the alternate STT/LLM/TTS providers, Simli/HeyGen
avatars, the CosyVoice/F5-Thai/FunASR spike servers, the echo-guard, all
browser-JS injection (avatar toggle, Android speaker toggle, loading overlay),
character mode, voice-expressiveness levers, and the phone/switch `.bat` scripts.
_(Character mode was later re-introduced in a cleaner form for the Thai validation demo — see the
top section.)_

**The stack now:** Deepgram STT → OpenRouter (Gemini Flash Lite) → ElevenLabs TTS
→ avatar, WebRTC → `/client`. The avatar is either the server-rendered **Ditto**
talking-head (`AVATAR=ditto`, default — see below) or **none** (audio-only;
client renders the face — the current focus, see the top section). `.env` knobs:
`LANGUAGE` (en/zh/th), `TTFO_TARGET_SECONDS`, `AVATAR`, `CHARACTER_MODE`. Verify
with `python -m scripts.preflight`.

## 🔧 Audit pass (2026-06-16): future-proofed against Pipecat drift
A maintenance sweep, no behavior change:
- **Deprecated Pipecat constructors migrated.** The three cloud stages built their
  services with soon-to-be-removed params (`live_options=`, `model=`, `voice_id=`);
  all now use the supported `settings=…Settings(…)` form. The LLM warmup's
  `_settings.model` path still resolves.
- **preflight now catches *soft* drift.** A new non-fatal **`DEPR`** status flags
  any `DeprecationWarning` raised *from our own code* (transitive stdlib/3rd-party
  ones are filtered out), so the next API drift is visible before it turns fatal.
- **Ditto server modernized:** deprecated `@app.on_event("startup")` → FastAPI
  `lifespan`; `asyncio.get_event_loop()` → `get_running_loop()`.
- Tidied the hot-path avatar logging and refreshed stale docs/comments.
- Live end-to-end run confirmed clean (all stages PASS, no warnings, exit 0).

## What this is
Real-time **speech → LLM → photoreal talking-head avatar**. Multi-turn,
streaming end-to-end. Goal: time-to-first-output **< 8 s**. Built on **Pipecat
1.3.0**, WebRTC to the browser.

## How to run
The avatar runs as a separate GPU server (its own `ditto` conda env). Start it
first, then the pipeline:
```
conda run -n ditto python -m local_services.ditto_server.app        # avatar server, :8002
python -m pipeline.main                                              # serves /client
```
For live, unbuffered server logs (useful while debugging), run the env python
directly instead: `E:\miniconda3\envs\ditto\python.exe -u -m local_services.ditto_server.app`.
Open `/client`, fully close the tab between tries (don't just refresh — sessions
are serialized server-side), wait for the avatar face, then talk.

## ▶️ Ditto avatar: integrated + GPU-realtime (live browser test in progress)
The avatar is now **Ditto** (antgroup/ditto-talkinghead, PyTorch path). Built:
- ✅ Repo + 13 GB checkpoints vendored under `local_services/ditto_server/vendor/`.
- ✅ `ditto` conda env (cloned from `musetalk`: py3.10 + torch 2.11+cu128 on the
  5060 Ti) + `onnxruntime-gpu`, `mediapipe`, `filetype`.
- ✅ **Compiler-free:** the one Cython kernel (`core/utils/blend`) was replaced
  with a NumPy equivalent, so no MSVC/CUDA compiler is needed.
- ✅ `local_services/ditto_server/` (FastAPI ws, `StreamSDK` subclass that diverts
  rendered frames to the socket) + `local_services/ditto_video.py` (mirrors the
  MuseTalk client contract). `pipeline/stages/avatar.py` builds `DittoVideoService`.
- ✅ Headless ws smoke (`ditto_server/ws_test.py`) → 512² RGB streaming.

**Bugs found + fixed during the live bring-up:**
- 🐛→✅ **Ran on CPU (laggy/desync).** onnxruntime's CUDA provider silently fell
  back to CPU (missing `cublasLt64_12.dll`). Fix: `app.py` adds torch's `lib/`
  (cu128 DLLs) to the DLL search path before onnxruntime loads → **GPU, ~24.9 fps
  (realtime is 25).** `DITTO_STEPS` (default 25) trades quality for headroom.
- 🐛→✅ **Avatar froze on reconnect.** The shared SDK re-ran `setup()` while the
  prior session's worker threads were still alive. Fix: a `_session_lock`
  serializes whole sessions (close before next setup) + a watchdog logs
  throughput/worker crashes.

**Lag + non-continuous motion: fixed (verify live).** Three distinct causes:
- 🐛→✅ **Frame-rate mismatch (drift/freeze).** Ditto's motion is locked to 25 fps
  but the WebRTC output played at 20 (`main.py`, a MuseTalk leftover) — 5 frames/s
  piled up in the live video queue, so video drifted behind audio then froze.
  Fix: `video_out_framerate=25`.
- 🐛→✅ **Bursty render (jerky motion).** The LMDM diffusion emits ~70-frame (2.8s)
  clips at the realtime edge (~24.9 fps, no headroom); the pump drained with no
  buffer, so each clip's compute-gap stalled then jumped. Fix: a jitter buffer in
  the server `pump()` (`DITTO_LEAD_FRAMES`, ~1s) + hold-last-frame over short gaps.
- 🐛→✅ **Neutral snap between sentences.** The pump showed the neutral portrait
  whenever the queue briefly emptied while `speaking` was clear — which toggles
  *per sentence*, so the face reset mid-turn. Fix: pump now returns to neutral
  only after `DITTO_IDLE_GRACE` of sustained empty queue.
- ✅ **A/V sync (frame-clocked) — replaced the fixed delay (2026-06-16).** The old
  `DITTO_AUDIO_DELAY_S` was a fixed wall-clock guess that drifted against the
  bursty render (the "avatar is delayed" symptom). Now the server's `pump()` emits
  `video_start`/`video_clock{frames}`/`video_end` markers (counting only *real*
  rendered frames) and the client (`ditto_video.py`) buffers the voice and releases
  it paced to those frames (`allowed_s = frames/fps + DITTO_SYNC_LEAD_S`) — so the
  voice **waits when the render stalls and never runs ahead**. `DITTO_SYNC_FALLBACK_S`
  (3 s) forwards the voice unsynced if no markers arrive (never goes silent), and
  barge-in flushes the server frame queue + the client voice buffer.
  - Also fixed: a **short turn** that renders fewer than `DITTO_LEAD_FRAMES` now
    primes (and emits markers) as soon as its speech ends, instead of stranding
    its frames below the priming threshold.
  - _Verified 2026-06-16:_ server markers stream correctly (smoke test: `video_start`
    + `video_clock` 5→70 at ~24.8 fps); browser `<video>` plays 512²; preflight
    clean. **Still needs a human check** for actual lip alignment — see below.
- ⚡ **TensorRT-FP16 acceleration (2026-06-17): render ~8 -> ~16 real fps, same visual.**
  The GPU is Blackwell (sm_120); the avatar was slow because the photoreal render ran in
  PyTorch. Built a no-compiler TensorRT path: `build_trt.py` compiles FP16 engines FOR THIS
  CARD from Ditto's ONNX graphs (numerically validated vs fp32 before use -- decode 1.25%
  err, rest <0.2%); `trt_runner.py` runs them via torch tensors (no cuda-python); `app.py`
  swaps decoder/stitch/appearance to engines behind **`DITTO_TRT`** (default on), warp +
  diffusion + aux stay PyTorch (warp needs the GridSample3D plugin = a compiler). Measured
  on the RTX 5060 Ti: **decode 41ms -> 14ms (2.9x)**, end-to-end render rate (watchdog
  `real_fps`, the true metric -- "fps received" is pump-padded) **~13.5 -> ~16 fps**. So the
  avatar now runs at **`DITTO_FPS=12`** (was 8) -- ~1.5x smoother, frame_q never starves, same
  photoreal face. _Honest limits:_ NOT full 25fps realtime -- the gate is the serial per-frame
  pipeline (warp+decode+stitch+putback ~62ms, GIL-bound); **warp (PyTorch ~21.5ms) is the
  remaining chunk**, accelerable only by building the GridSample3D TRT plugin (needs the
  VS+CUDA compiler). Diffusion steps don't change real_fps; LMDM-TRT gave no net gain (numpy
  sampling loop's host roundtrip cancels it) so it's not wired in. Set `DITTO_TRT=0` for the
  pure-PyTorch fallback (then DITTO_FPS=8). Run `build_trt.py` once per machine (engines are
  GPU-arch-specific, cached in `checkpoints/ditto_trt_blackwell/`).
- 🔄 **A/V sync + "avatar is delay" diagnosis (2026-06-17, in progress).** After TRT, the avatar
  still felt delayed. What was done + learned:
  - **fps unified to 12** across server (`DEFAULT_FPS`), client (`ditto_video.py` reads `DITTO_FPS`),
    and transport (`main.py video_out_framerate`). `main.py` was defaulting to 8 vs server 12 = a real
    drift bug, now fixed.
  - **Jitter buffer** `DITTO_LEAD_SECONDS` (default 0.5 -> primes ~0.5s of real frames before
    `video_start`) absorbs render bursts so playout doesn't starve.
  - **`sync_with_audio` mode** (`DITTO_SYNC_WITH_AUDIO`, default 1): `ditto_video.py` buffers each
    turn's real frames and releases them tagged `OutputImageRawFrame.sync_with_audio=True` right after
    the matching voice, so pipecat shows each frame at its audio position (transport audio queue, not an
    independent video clock). Fixes A/V DRIFT in the browser.
  - **WebRTC fix (machine-level, applied via elevated cmd):** disabled Teredo + added a Windows
    Firewall allow rule for the pipeline python -- ICE was failing (no video) until this.
  - **`local_services/ditto_server/capture_mp4.py`** (new): records the LIVE server stream to an mp4
    (`--lead` shifts video to test lip offset) -- lets you judge the avatar WITHOUT WebRTC/RDP.
  - **Diagnosis:** the render IS smooth; the persistent "delay" is **Ditto's diffusion WARMUP** (needs
    `valid_clip_len = 80 - overlap` ~2.2s of audio before the FIRST frame) + **subtle lip motion**
    (measured lip-vs-audio corr only ~0.3 via `scripts/avatar_tune.py align`). A ±0.5s lip shift was
    imperceptible to the user -> it's NOT a tunable offset, it's the warmup. This is fundamental to
    Ditto on this GPU.
  - **DECISION: keep Ditto** (user's call over switching to MuseTalk, which has no warmup + sharper
    lip-sync but a different mouth-on-photo look).
  - **NEXT (Part B):** (1) install VS BuildTools + CUDA Toolkit 12.8 (keep driver 591.44), build the
    **GridSample3D plugin** -> TRT the **warp** (~21.5->~7ms) -> render ~16->~20+fps. (2) **Cut the
    warmup** (the real delay): raise `DITTO_OVERLAP` (smaller valid_clip_len = less audio before first
    frame; 45->1.4s, 60->0.8s vs 25->2.2s), compensating the intrinsic lip-lead via `DITTO_SYNC_LEAD_S`.
    Tools ready: `capture_mp4.py` (judge offline), `_fps_probe.py` (sustained fps), `avatar_tune align`
    (lip offset). Honest note: he's on **RDP**, which adds its own choppiness -- judge on the machine's
    own monitor; and trim mp4s with re-encode (not `-ss -c copy`, which breaks the start).
- ✅ **FINAL realtime config (2026-06-17): live @ 8fps, overlap 25, no trim.** After the
  delay/freeze/truncation thrash, the harness pinned the working point by measurement:
  **live streaming (`DITTO_PRERENDER=0`), `DITTO_FPS=8`, `DITTO_OVERLAP=25`.** Why each:
  (1) **fps 8** is the rate this GPU SUSTAINS at overlap 25 — measured `real_fps` 8.0–8.5 with
  `frame_q` never starving, so the playout buffer can't empty → **no freeze** (the freeze was
  render starvation at fps>sustained). (2) **overlap 25** has a measured intrinsic lip offset of
  **0.00s** (cleanest of all windows) → lips locked, `DITTO_SYNC_LEAD_S=0`. (3) **removed the
  tail-trim + queue-clear** from `pump()` — on a sub-realtime GPU the next sentence renders into
  the queue before the current finishes, so the clear was discarding it (the "avatar doesn't
  speak the whole sentence" truncation). Now frames drain continuously, voice frame-clocked →
  synced, never cut; the silent tail just plays (acceptable). (4) **`video_out_framerate`
  coupled to `DITTO_FPS`** in `main.py` (was hardcoded 25 → 3× mismatch = desync). Verified
  headless on the 3-sentence reply: coverage ≥1.5 every sentence (no truncation), `real_fps`
  sustains 8 (no freeze), 0.00s offset. Residual = the GPU floor: ~2–3s warmup before the first
  frame, choppier 8fps motion, and the over-VPN WebRTC path (Tailscale ICE failures) is separate.
- ✅ **The lag's root cause + the real fix: frame-dropping to ~12.5 fps (2026-06-17).**
  Instrumenting `real_fps` (the *actually-rendered* frame rate, vs the neutral-padded
  `sent`) proved this GPU renders the 512² photoreal avatar **below the native 25 fps**,
  so at 25 fps the video can never keep up — hence the persistent "voice first, avatar
  late" drift, regardless of resolution or diffusion steps. Fix: **render fewer frames.**
  `DITTO_FPS` (default **12.5**) is the output rate; below 25 the server drops the
  in-between motion frames via a `_StridedQueue` on the motion→render handoff — *before*
  the costly warp/decode/putback — so the GPU does half the work and **keeps up in real
  time** (measured ~13 fps real ≥ 12.5 target → zero drift). Sync is safe because A/V is
  frame-clocked either way; lower fps just means choppier motion, not desync (the motion
  is relative to a fixed `d0`, so subsampling is correct). Paired change: `DITTO_OVERLAP`
  default → **10** (wide 70-frame diffusion window = cheapest diffusion = best `real_fps`).
- ✅ **Final realtime sync mode: compute-first buffer + frame-drop + tail-trim (2026-06-17).**
  Frame-drop alone fixed *drift* but the log still showed two residual desyncs on short
  sentences: a **late start** (lips begin ~1.7s after the voice — live mode waits for
  `LEAD_FRAMES`, which lands after the short sentence's voice is fully buffered) and a
  **silent tail** (the full-window end-of-sentence flush renders ~3s of closed-mouth
  frames that then *played*, so the mouth moved after the voice ended). Fix = combine both
  mechanisms: **`DITTO_PRERENDER` default → 1** (compute-first: wait until the sentence is
  fully rendered, then drain it with the voice frame-clocked to it → `video_start` and the
  voice release together, so the start can't lag) kept **alongside** frame-drop (so the
  per-sentence pre-render is fast enough to sustain). Plus a **tail-trim** in `pump()`: each
  turn captures `turn_frames = round(expected)` (the voice-frame count, from real audio
  samples / 640 / `frame_stride`) and stops + discards the queue once those have played, so
  the silent flush frames never show. Cost: a ~3-4s render-wait before each short sentence
  appears (the inherent "buffer-first" tax on a sub-realtime GPU; ~half the old 25fps
  compute-first wait because we render half the frames). _Verified headless 2026-06-17;
  needs the human lip-alignment check below._

- ✅ **Lip-sync root cause found + fixed by measurement (2026-06-17).** "Lips don't match
  the voice" persisted even after the timing fixes because there was a **constant A/V offset
  the `hold`/`tail` metrics never checked**: the render's mouth motion *leads* the audio by a
  fixed time (a model look-ahead). Built `scripts/avatar_tune.py` — an autonomous harness that
  drives the Ditto ws server with reproducible speech clips, keeps the returned RGB frames, and
  cross-correlates a **mouth-region motion envelope vs the audio-RMS envelope** to measure the
  intrinsic lip offset (the thing `hold` can't see). Findings (avatar_tune `align`, 18s clip,
  reproduced): the lead is a **fixed time** (−0.40s at fps12.5 == −10 frames at fps25) and is
  set by `DITTO_OVERLAP` — overlap 10 → 0.40s lead (corr 0.20); **overlap 45 → 0.18s lead (corr
  0.29, cleanest)**. The old default (overlap 10, lead 0.2) left a ~0.2s residual desync. Fix:
  **`DITTO_OVERLAP` default 10 → 45** (smaller, cleaner intrinsic lead; in compute-first the
  finer window doesn't gate playback), which the existing `DITTO_SYNC_LEAD_S=0.2` cancels to a
  **net ~0.04s** (validated by re-measuring the baked default). Also tightened the `video_clock`
  release granularity **5 → 2 frames** to cut the 0.4s sawtooth jitter in voice release. Full
  write-up + the per-config table in `output/avatar_tuning_report.md`. _Latency/smoothness
  deliberately deferred to a next round (user: "lip-sync first")._

- ⚡ **Perf round 2 (2026-06-18): per-stage profiling overturned the bottleneck; putback fixed.**
  Added per-stage timers (`app.py` `_timed`/`_profile_attach`, `DITTO_PROFILE` default on) and measured
  the per-frame cost at full load (stride 1) -- this **corrects the STATUS assumption that warp was the
  gate**:
  - **putback ~75ms (CPU) -- the single biggest cost**, warp ~47ms (PyTorch GPU), decode ~22ms (TRT),
    stitch ~2ms. The pipeline gate = max(GPU~66ms, CPU putback~75ms) -> ~13-15 real fps.
  - Root cause of putback: it alpha-blended the 512 render over the **entire 1536x1024 source portrait
    in float64** every frame (`core/atomic_components/putback.py`), then the frame was downscaled to 512
    for output. **Fix: composite only the render's destination bbox, in float32** (lossless -- the rest
    of the frame is unchanged). Validated numerically vs the old full-frame blend (max abs diff 7,
    0.13% of pixels differ by >1 == mask-edge rounding).
  - **Result: real_fps ~14 -> ~21-22** (full load, stride 1), measured + a clean `capture_mp4` of 698
    frames. **No compiler, no new deps.** The renderer now sustains ~21fps, so `DITTO_FPS` can be raised
    12 -> ~16-18 for a visibly smoother avatar (one env var; all three fps sites read `DITTO_FPS`).
    Needs the live lip-sync check before committing the higher fps.
  - **Now co-gated by warp (47ms GPU) + putback (46ms CPU).** Next levers: (1) **TRT the warp**
    (GridSample3D plugin, needs the VS+CUDA toolchain = Part B) -> warp ~47->~15ms -> ~30fps potential;
    (2) GPU-resident render chain (no deps) to trim the decode/warp host round-trips.
  - Side bug found: a dirty client disconnect can leak `_session_lock` (a websockets teardown
    AssertionError on the send path), wedging the server until restart -- likely the "froze on reconnect"
    class. Small robustness fix pending.

- ⚡ **Perf round 3 (2026-06-18): TRT the warp via a self-built GridSample3D plugin -> ~25 fps.**
  Installed the compiler toolchain (VS BuildTools MSVC 14.44 + Windows SDK 10.0.26100 + CUDA
  Toolkit 12.8.93, **driver kept at 591.44** -- toolkit-only install, no Display.Driver) and built
  the missing piece: the **GridSample3D TensorRT plugin** the warp ONNX needs (`grid_sample` 5D op).
  - Source `grid-sample3d-trt-plugin` (registers `GridSample3D`), ported to **Windows + TRT 10.16 +
    sm_120** (only change needed: `CUDA_ARCHITECTURES=120`, real include/link dirs; the
    `IPluginV2DynamicExt` API still compiles on TRT 10). Build staging in
    `local_services/ditto_server/plugin_build/` (plugin source, TRT 10.16 headers from OSS, and a
    **synthesized `nvinfer.lib`** import lib made from the runtime DLL -- no gated SDK download).
    Built `grid_sample_3d_plugin.dll` -> staged next to the engines in `ditto_trt_blackwell/`.
  - Built `warp_network_fp16.engine` (parser resolves GridSample3D with the plugin loaded);
    **validated 0.49% mean err vs the PyTorch warp** (max abs 0.0045).
  - Wired in `app.py`: `_load_grid_sample_plugin()` (CDLL + init_libnvinfer_plugins, best-effort)
    + warp added to `TRT_SWAPS`, gated on the plugin (falls back to PyTorch warp if the DLL is
    absent). `trt_runner.py` deserializes the warp engine fine once the plugin is registered.
  - **Result: warp ~47 -> ~26ms; real_fps ~21 -> ~25-26 (frame_q grows at the 25 drain -> ceiling
    >= 26), GPU now peaks ~95%.** The avatar renders at **full native 25 fps** in real time. Clean
    1541-frame `capture_mp4` at fps 25 (`output/realtime_capture_warp_trt.mp4`). **`DITTO_FPS` can
    now be raised 12 -> ~20-25** (one env var) for max smoothness -- pending the live lip-sync check.
  - **Cumulative (round 2 + 3): real_fps ~14 -> ~26 (~1.85x), no quality change.** Remaining
    per-frame: decode ~24ms (TRT) + warp ~26ms + putback ~36ms (CPU, overlapped). Further headroom
    would come from the GPU-resident chain (cut host round-trips) or FP8 decode -- not needed to hit
    25 fps, so deferred.
  - Profiling note: `DITTO_PROFILE` (per-stage timers via `_timed`/`_profile_attach`) is now
    **default OFF** and only wraps pure callables (warp/decode/putback, idempotent) -- it must NOT
    wrap `motion_stitch` (the SDK calls `.setup()` on it; doing so crashed the 2nd session).

- 🔬 **Warmup research + idle-mask (2026-06-18): the "2.8s wall" is half real; mask it, don't
  break sync.** Researched FFD (First-Frame Delay = the warmup: time from a turn's audio start to
  the first lip frame). Ditto's paper reports **385ms** FFD and says **higher overlap_v2 -> LOWER
  FFD** (less audio lookahead: `valid_clip_len = 80 - overlap_v2`); the repo ran overlap 25 (~2.2s).
  Ditto also ships an unused **online preset** (`v0.4_hubert_cfg_trt_online.pkl`): **overlap 70,
  steps 10** (the `v_min_max_for_clip` motion clamp + `fix_kp_cond=1` are ALREADY inherited via
  default_kwargs -- the round-2 "no clamp" guess was wrong, verified by dumping the pkls). Measured
  the levers at fps 12 (warp-TRT headroom now lets them keep up -- the round-2 throughput blocker
  is gone):
  - **overlap 70 / steps 10:** FFD 2.84 -> ~1.1s, silent tail 2.5 -> 0.83s, real_fps holds 12 --
    BUT lip-sync goes **erratic** (intrinsic offset estimates swing -0.75s..+0.92s across align/calib;
    NOT a constant `DITTO_SYNC_LEAD_S` can fix). overlap 45 better (+0.33s) but still drifted.
  - **steps 25 -> 10 at overlap 25** ALSO broke sync (0.00 -> -0.67s). So **overlap 25 / steps 25 /
    lead 0.0 is a co-calibrated sweet spot** -- perturbing EITHER knob destabilizes the mouth. The
    FFD-vs-sync tension is real; the round-2 "wall" was right about *sync*, wrong to call FFD fixed.
  - **DECISION: keep the sync sweet spot, MASK the warmup.** Measured that feeding silence renders a
    LIVING resting face (blinks + micro-motion, eye-motion ~4.6 w/ spikes to 16), so during the
    per-turn priming gap we now play a looped idle clip instead of a frozen portrait -- the warmup
    reads as natural listening. **Implemented in `app.py`** (no diffuser-knob change, sync untouched):
    after d0 warmup, render ONE idle window (frame-dropped via the early `frame_stride`), drain to
    quiescence (stragglers must NOT leak into turn 1), cache on `engine.idle_frames` (same portrait
    every connect -> captured ONCE per server life, ~+2.7s on the first connect only, reused free
    after), and `pump()` ping-pongs that loop during priming/idle. Knob: `DITTO_IDLE_CAPTURE_CHUNKS`
    (0=auto/on, <0=off). Verified: lip-sync still **+0.00s** (align), realistic setup-then-realtime
    multi-turn coverage 1.5-2.25 (clean, no leak), idle face LIVING during priming, cache reused.
    Offline artifact: **`output/realtime_idle_mask.mp4`** (start_latency 2.46s, 698 frames).
  - _Harness caveat:_ `avatar_tune live`/`measure` RESTART the server and dump audio DURING the
    ~5s first-connect capture, so the socket buffers it mis-paced -> bogus FFD/coverage with capture
    ON. Judge capture-ON via `align` (sync) + a realistic setup-then-realtime probe, not `live`.
  - _Residual / not solved:_ FFD itself is still ~2.2-2.8s (masked, not removed). Truly cutting it
    needs a model with stable high-overlap sync (e.g. the REST paper's distilled student, arXiv
    2512.11229) or fewer-step sampling that doesn't move the lips -- a model change, not a config.

- 🎯 **A/V SYNC ROOT CAUSE FOUND + FIXED (2026-06-18): `sync_with_audio` was DEAD under the
  live transport.** The persistent "voice and avatar not in sync" was never a server/overlap
  problem -- every prior round tuned the Ditto server, which measures **0.00s** intrinsic offset
  (`avatar_tune align`, `capture_mp4`). The desync is **one layer downstream, in the pipecat
  transport**, and had never been instrumented. Found by reading `pipecat/transports/base_output.py`:
  - `DittoVideoService` tags each turn's frames `OutputImageRawFrame.sync_with_audio=True`
    (`DITTO_SYNC_WITH_AUDIO`, default on) to pin each frame to its audio. The transport routes a
    tagged frame through the AUDIO queue and renders it via `_video_images` -- **which is ONLY read
    when `video_out_is_live` is FALSE** (the non-live `_video_task_handler` branch, line 895).
  - But `main.py` set **`video_out_is_live=True`** (needed so the idle face animates). Under is_live
    the video task runs `_video_is_live_handler`, which reads ONLY `_video_queue` and NEVER
    `_video_images`. So **every lip-synced frame was silently dropped**; only untagged frames (idle +
    the unsynced fallback) animated, on a FREE-RUNNING 12fps clock that self-resets on stalls
    (line 916). Two independent clocks (audio paced to render markers, video free-running) = the drift.
    The whole `sync_with_audio` "A/V lock" added 2026-06-17 was a **no-op the entire time.**
  - **Fix (`main.py`):** the two modes are mutually exclusive in pipecat 1.3.0, so couple them off the
    same flag -- `video_out_is_live = want_video and not DITTO_SYNC_WITH_AUDIO`. Sync on -> NON-live
    transport -> `_video_images` IS read -> turn frames display **paced through the audio queue (true
    per-frame A/V pinning)**; idle frames animate via `_set_video_image`. Sync off -> is_live True
    (legacy free-running clock). One-line revert: `DITTO_SYNC_WITH_AUDIO=0`.
  - **Verified WITHOUT a browser:** `scripts/_sync_routing_test.py` drives pipecat's real
    `MediaSender` with a fake transport recording what gets DRAWN -- is_live=True draws `['idle']`
    (synced frame dropped = the bug), is_live=False draws `['idle','turn']` (synced frame drawn = the
    fix, idle still drawn = no freeze). Preflight clean. **Still needs the human live check** (only the
    browser shows true lip alignment) -- but the mechanism is now correct, not a free-running clock.

- 📶 **Remote viewing: jitter buffer injected server-side (2026-06-18).** Judging the avatar over
  RDP or a remote Tailscale link is unreliable -- RDP carries video+audio on separate paths (can
  desync), and a remote viewer hits WAN jitter. Measured: a notebook on the tailnet connects over
  the **internet** (public IP, ~110ms ping with spikes to 316ms), while the box renders a steady 12fps
  with the GPU ~96% idle -> the lag is the NETWORK, not the render. Fix = bigger receive-side WebRTC
  jitter buffer. Done **server-side so every device gets it** (no per-device console tweak):
  `pipeline/main.py::_install_client_jitter_buffer()` adds FastAPI middleware that injects a tiny
  `<script>` into the served `/client` index (prebuilt bundle untouched) which patches
  `RTCPeerConnection` so each receiver sets `jitterBufferTarget` (Chromium) / `playoutDelayHint`
  (legacy). Knob: **`CLIENT_JITTER_BUFFER_MS`** (default 400; 0 disables). Verified the script is in
  the page over both localhost and the HTTPS Tailscale URL, bundle intact. Tradeoff: smoother but
  +~400ms latency; if the link is bandwidth-starved (freezes, not jitter) lower `DITTO_FPS`/size
  instead. Truly jitter-proof remote option = audio-only mode (`AVATAR=none`, face rendered on the
  client, only voice on the wire). **Viewing setup:** `tailscale serve` is live ->
  `https://porsche-pc.tail21bb8a.ts.net/client` (HTTPS = mic works; localhost binding is fine since
  serve proxies locally). Same-LAN is the lowest-jitter path if the notebook can join this PC's network.
  - **Tuning outcome (live, Taiwan notebook <-> Thailand box):** buffer OFF -> both A/V stutter
    (the link genuinely has jitter); 400ms -> audio smooth but avatar trails the voice; settled the
    user at **`CLIENT_JITTER_BUFFER_MS=600`** (their call -- prioritize smooth audio, accept the lip
    lag). Same-LAN and audio-only were both ruled out (different countries; client-render not an
    option), so buffer tuning is the only lever left short of lowering `DITTO_FPS`/frame size.

- ✅ **Remote lag SOLVED by fit-the-stream + a live isolation test (2026-06-20).** The remote
  avatar lag (Thailand viewer) was re-attacked with a clean isolation test, then fixed by moving
  the LIVE `.env` to the fit-the-stream config -- which had only ever been *documented*, never set
  (the `.env` was still running 512px + a 600ms buffer, i.e. the pre-fix state).
  - **Isolation test (`scripts/stream_live.py`, new):** streams a pre-rendered avatar mp4 LIVE as
    MJPEG (ffmpeg `-re`, real-time paced, NO client buffering, NO GPU render), exposed over the
    existing `tailscale serve` on a side path (`--set-path /watch`). The client renders frames to a
    `<canvas>` via fetch + multipart parse -- NOT `<img src=multipart>`, which Safari/some browsers
    silently ignore (-> black screen, a real dead end hit twice). Isolates link-vs-system without
    WebRTC/RDP noise. (Sibling `scripts/serve_clip.py` = the earlier buffered-file version, kept.)
    - Result (live, Taiwan box -> Thailand notebook): **512px @ ~2Mbps -> 18fps + constant lag;
      320px @ ~1.2Mbps -> 24fps, keeps up.** The link carries ~1.2Mbps live fine but saturates
      near ~2Mbps; the render was never the gate (frames are produced fine).
  - **KEY FINDING -- bandwidth is NOT the hard limit (overturns "smaller is always better").**
    Pushing the avatar ULTRA-low (160px / 200kbps) was *choppier* than 256px / 400kbps: a 200k cap
    starves the VP8 software encoder (can't fit even a tiny frame + periodic keyframes at a steady
    12fps, so it drops frames). Smoothness needs ENOUGH bitrate for steady frames, just under the
    link ceiling (~1.5Mbps). So do NOT shrink-and-starve; moderate frame + comfortable cap wins.
  - **Fix = live `.env` values** (the `main.py` bitrate-cap + jitter-buffer code was already active
    by default; only the live values were wrong). **Accepted config (user, 2026-06-20):
    `DITTO_SIZE=256`, `WEBRTC_VIDEO_BITRATE_MAX=400000`, `CLIENT_JITTER_BUFFER_MS=600`** -- "quality
    and smoothness acceptable for now". Headroom remains to climb quality (320/384) or trim the
    600ms buffer for less delay; left at the accepted point.
  - **`scripts/run.ps1` hardened (the silent-512 root cause):** the Ditto server (`app.py`) reads
    `DITTO_SIZE` from the OS env ONLY -- it does NOT load `.env` (no `python-dotenv` in the `ditto`
    conda env). So editing `.env` alone left the server at its 512 default while the pipeline used
    the new size = a silent mismatch. run.ps1 now parses `DITTO_SIZE` from `.env` and exports it so
    BOTH child processes agree; `.env` is the single source of truth again.
  - **Gotcha:** open the client at **`/client/` WITH the trailing slash** -- the prebuilt page
    references assets relatively (`./assets/...`), so `/client` (no slash) resolves them to
    `/assets/...` -> 404 -> white screen.
  - Supersedes the 2026-06-18 "settled at 600ms, the only lever" conclusion above: frame size +
    bitrate cap are the bigger levers; the buffer is now a latency<->smoothness trim, not the fix.

- 🔊 **TTS fallback: ElevenLabs out of credits -> Deepgram Aura (2026-06-18).** Live symptom was
  "avatar shows but won't talk back, in both chat and voice." Root cause (caught by a direct API
  test, not the avatar/network/sync): the **ElevenLabs account was out of credits** (1 of 10000
  left) -> TTS returned no audio (the logged 33s TTFB was the websocket hanging on the quota error),
  so every reply died at the TTS stage. STT->LLM were verified working in the logs (heard "hello",
  replied "Hi there! How can I help you today?"). **Fix:** added a single TTS-provider switch --
  `TTS_PROVIDER=deepgram` builds **Deepgram Aura** (`aura-2-helena-en`, reuses `DEEPGRAM_API_KEY`)
  in `pipeline/stages/tts.py`; ElevenLabs stays the code default. English-only, so it doesn't cover
  the zh/th character modes. Verified: direct Aura call returned 87KB audio in 2.4s; `build_tts()`
  returns `DeepgramTTSService` under the live `.env`. **Revert** by removing `TTS_PROVIDER` once
  ElevenLabs is topped up. (Note: this is a deliberate single fallback switch, not a return to the
  removed multi-provider branching.)

**Open / next:**
- **Live lip-sync tuning (human check).** Headless tests can't judge lip alignment,
  and the fake-mic test injects constant VAD interruptions (which reset the render
  before it primes → fallback). Open `/client` with a **real mic + headphones**,
  speak a multi-sentence turn, and tune `DITTO_SYNC_LEAD_S` (raise = voice later,
  lower/negative = voice earlier) until lips track the voice. **Also judge the new idle
  mask:** during each turn's ~2.2s warmup the face should now blink/breathe (a living
  listen), not freeze. `output/realtime_idle_mask.mp4` shows it offline (no RDP noise).
- **Text/voice mismatch** — likely the removed echo-guard (self-talk on speakers;
  see tradeoffs). Confirm from logs; fix = re-add the guard or use headphones.

Once the live turn is clean → **Phase C:** remove the MuseTalk server +
`musetalk_video.py` for the Ditto-only end state.

## ⚠️ Known tradeoffs (accepted in the cleanup)
- **Echo-guard removed** → the avatar can self-talk again (mic hears its own
  voice / STT-tail). Use headphones; the guard is recoverable from git history.
- **Azure zh path removed** → Deepgram + ElevenLabs still do zh-TW, but Azure was
  the higher-quality Asia-region option; re-add locally if zh quality disappoints.
- **No avatar fallback** → Ditto-only; the fallback is git history.

## Key files
- `pipeline/main.py` — pipeline assembly, LLM warmup, greeting; audio-only branch
  (skips the avatar stage + server video) and the `EmotionTagger` insertion.
- `pipeline/config.py` — keys + `LANGUAGE` (en/zh/th), `AVATAR`, `CHARACTER_MODE`
  (incl. the in-character Thai system prompt), driven by `.env`.
- `pipeline/stages/*.py` — per-stage factories (stt/llm/tts/avatar/vad).
- `pipeline/stages/emotion_tagger.py` — extracts the leading `[emotion]` tag, strips
  it before TTS, pushes it to the client (`RTVIServerMessageFrame`). Character mode only.
- `pipeline/metrics.py` — `TtfoMeter` (logs `[TTFO]` per turn + summary).
- `local_services/ditto_server/app.py` — Ditto GPU server (ws, frame interception,
  GPU DLL fix, session lock, watchdog, frame-clock `pump()` markers); `vendor/`
  holds the repo + checkpoints.
- `local_services/ditto_video.py` — Pipecat client for the Ditto server; owns the
  frame-clocked A/V sync (buffers the voice, releases it paced to rendered video).
- `local_services/musetalk_server/` + `musetalk_video.py` — old avatar, removed in
  Phase C (kept until Ditto's live test is clean).
- `scripts/preflight.py` — import/drift check.
