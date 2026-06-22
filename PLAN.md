# VisualLLm — Full Project Plan & Progress

_Last updated: 2026-06-10 (historical plan — see banner)_

> **⚠️ Partly historical.** The current stack + state live in **`STATUS.md`** (source of
> truth). Since this plan was written: avatar moved Simli → **MuseTalk** (default; Ditto
> kept as fallback), TTS moved ElevenLabs → **CosyVoice2 on vLLM in WSL** (TTFB ~1.1s),
> and the remote-mic / log-flood / lip-sync reliability issues were fixed (STATUS.md,
> 2026-06-22). Read STATUS.md first; the phase narrative below is kept for history.

## 👉 WHERE WE ARE RIGHT NOW
**Phase 1 (English) is DONE, and Phase 3's local MuseTalk avatar is now
IMPLEMENTED and working on the 5060 Ti** — the cloud-avatar lag from Thailand is
fixed. The local avatar streams lip-synced frames at **~1.14× realtime (20 fps)**.
Remaining: the user's **visual check in the browser** (lip-sync + audio/video
sync) and replacing the demo portrait with their own face. Phase 2 (Mandarin) is
intentionally deferred.

```
Phase 0  Scaffold ............................ ✅ DONE
Phase 1  English prototype (working demo) ..... ✅ DONE   ← measured TTFO ~2s
Phase 2  Mandarin (zh-TW) swap ................ ⬜ DEFERRED (user's call)
Phase 3  Local MuseTalk avatar on 5060 Ti ..... ✅ IMPLEMENTED — needs visual check
Phase 4  Conversation polish .................. 🟡 PARTIAL (fallback warning added)
```

### Phase 3 — how it was built (the non-obvious parts)
- **Separate `musetalk` conda env** (py3.10, **cu128 torch 2.11** — the only build
  that drives the Blackwell 5060 Ti; MuseTalk's pinned torch 2.0.1 cannot). Kept
  apart from the pipeline env, which it only talks to over the `:8002` websocket.
- **No mmpose/mmcv** (this machine has no CUDA compiler/MSVC, so DWPose can't
  build). DWPose was only used for the 68 iBUG face landmarks — replaced with
  **`face_alignment`** (pip-only, pure-torch) feeding MuseTalk's exact bbox math.
  Landmarks run once during avatar prep; the realtime loop never needs them.
- Server: `local_services/musetalk_server/app.py` (upstream cloned in `vendor/`,
  weights in `vendor/MuseTalk/models`). Streams PCM→frames; avatar materials are
  cached after first prep. Client (`musetalk_video.py`) resamples TTS audio to
  16 kHz for Whisper. Run it in its env: `run_server.bat` (or the conda command).
- Perf on the 5060 Ti: unet+vae ≈ 32 ms/frame (GPU floor) → 20 fps with headroom.
  Fixes applied: cap base portrait to 768 px (compositing was the bottleneck),
  cudnn autotune + TF32, `weights_only=False` shim, `TORCHDYNAMO_DISABLE=1`.

### ⚠️ Known thing to verify (user, in browser)
Audio is forwarded downstream immediately while video lags by the render time, so
**lips may trail the voice by ~0.5–0.8 s**. If that's visible, the fix is to delay
audio in `musetalk_video.py` to emit it paired with each video frame (costs a bit
of TTFO, well within the 8 s budget). Also **replace `assets/avatar.png`** (the
MuseTalk demo face) with your own front-facing portrait — delete
`local_services/musetalk_server/avatar_cache/` after to force re-prep.

---

## 1. The goal
Build a system where you **speak**, an LLM answers, and a **photoreal 2D talking
head avatar speaks the answer** (lip-synced audio + video). Multi-turn, with
interruption (barge-in). Everything streams.

**Hard target:** time-to-first-output (you stop speaking → avatar starts
responding) **< 8 seconds**. ✅ Currently ~2 s.

**Languages:** English first (prototype) → **Mandarin / zh-TW** (real target).
**Compute:** RTX 5060 Ti (16 GB) local + cloud APIs (hybrid).
**Reality:** running on a remote PC in **Thailand** via RDP.

## 2. The pipeline
```
speech → STT → LLM → TTS → lip-sync avatar → audio+video → browser
```
Built on **Pipecat 1.3.0** (handles streaming + barge-in), WebRTC to the browser.
Every stage is swappable from `.env` (no code change).

| Stage | Now (Phase 1) | Mandarin / local target |
|-------|---------------|--------------------------|
| Voice detection | Silero VAD (local) | same |
| STT (speech→text) | Deepgram | Azure (zh+Asia) or FunASR/Whisper local |
| LLM (brain) | OpenRouter → Gemini 2.5 Flash Lite | Qwen / DeepSeek; or local Qwen |
| TTS (text→speech) | ElevenLabs (voice "Adam") | Azure zh / MiniMax; or CosyVoice2 local |
| Avatar (lip-sync) | Simli (US cloud) | **MuseTalk local on 5060 Ti** |
| Transport | WebRTC → browser `/client` | same |

## 3. Why it can hit < 8 s — streaming
Nothing waits for a full step. The first LLM sentence flows to TTS → first audio
chunk → avatar starts talking, all while the LLM is still writing. Measured
budget on the working system:

| Stage | Measured |
|-------|----------|
| LLM first token | ~0.7–1.4 s (Gemini Flash Lite) |
| TTS first audio | ~0.15 s (ElevenLabs) |
| Avatar / STT / VAD / transport | the rest |
| **End-to-end TTFO** | **median 1.97 s, p95 2.86 s** ✅ |

---

## THE PHASES

### Phase 0 — Scaffold ✅ DONE
Repo structure, `.env`-driven config, per-stage factories, metrics, preflight
check. All syntax-validated.

### Phase 1 — English prototype ✅ DONE
Goal: a working talking-head demo, latency under 8 s.
- Installed Pipecat 1.3.0 + provider SDKs; migrated all code to the 1.3.0 API.
- Wired Deepgram + OpenRouter(Gemini Flash Lite) + ElevenLabs + Simli.
- Got it running over WebRTC → `/client`, accessed via RDP from Thailand.
- **Measured TTFO ~2 s — goal met.**
- Extras added: LLM connection **pre-warming**, and a **loading overlay** on the
  UI that waits for the avatar's video before clearing.
- Known limit: Simli (US cloud) has a ~10–15 s warmup + stutter from Thailand.

### Phase 2 — Mandarin (zh-TW) swap ⬜ NEXT — the research goal
Goal: the system works in Traditional Chinese. Mostly a `.env` change.
- `LANGUAGE=zh`
- LLM → a strong Chinese model (Qwen / DeepSeek via OpenRouter)
- TTS → a Mandarin voice (ElevenLabs multilingual, or **Azure zh** — already wired)
- STT → keep Deepgram zh-TW, or switch to **Azure** (better zh + Asia region =
  lower latency from Thailand)
- Test with native speakers; re-measure latency.

### Phase 3 — Local MuseTalk avatar on the 5060 Ti ✅ IMPLEMENTED
Goal: remove the cloud dependency that caused avatar warmup/stutter/failures from
Thailand. **Done** — the gray-frame stub is now a real streaming MuseTalk server.
- `AVATAR_PROVIDER=musetalk_local` is set in `.env`; `MUSETALK_URL=:8002`.
- Two processes: the **pipeline** (system python) + the **MuseTalk server** (the
  `musetalk` conda env). Start the server first (`run_server.bat`), then the
  pipeline. The server holds ~3–4 GB VRAM; keep the LLM on API (don't also enable
  `qwen_local`) so 16 GB is enough.
- Verified headless: engine smoke test (realtime), websocket round-trip (correct
  frames), `/health` ok, pipeline boots + serves `/client`. Needs the browser
  visual check (see top of this file).
- Optional later: TTS → CosyVoice2 local, STT → FunASR local (Phase 2/3 wiring
  already present).

### Phase 4 — Conversation polish ⬜ TODO
Barge-in tuning, multi-turn memory, error/timeout fallbacks (API fails → local
backup), and a small latency dashboard.

---

## Immediate to-do (start of next session)
1. **Visual check the local avatar** in the browser: start the MuseTalk server,
   then the pipeline, open `/client`, hard-refresh, speak — confirm the face
   lip-syncs and judge audio/video sync (see the ⚠️ note at the top).
2. **Replace `assets/avatar.png`** with your own front-facing portrait, then
   delete `local_services/musetalk_server/avatar_cache/` to force a re-prep.
3. **Rotate the OpenRouter key** (it was briefly in `.env.example`).
4. (Deferred) **Phase 2 (Mandarin)** when you want it — a `.env` swap.

(See `NEXT_STEPS.md` for the plain checklist, `STATUS.md` for technical detail.)

## How to run (local MuseTalk avatar = two processes)
```
cd E:\Claude\VisualLLm
# 1) MuseTalk avatar server — in its own conda env (start this FIRST):
local_services\musetalk_server\run_server.bat
#    (or: conda run -n musetalk python -m local_services.musetalk_server.app)
# 2) The pipeline — in the normal env:
python -m pipeline.main          # → http://localhost:7860/client
python -m scripts.preflight      # check all keys/imports resolve
python -m scripts.bench_latency --stage llm   # measure a stage
```
To go back to the cloud avatar, set `AVATAR_PROVIDER=simli` in `.env` (no server
needed). MuseTalk env/weights live in `local_services/musetalk_server/` (gitignored).
All settings live in `.env` (gitignored). Original plan file:
`C:\Users\MARU\.claude\plans\i-come-to-taiwan-gleaming-sparrow.md`.
