# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **`STATUS.md` is the source of truth** for current state and what's in
> progress — read it first. **`WORKFLOW.md`** is the detailed end-to-end workflow
> (turn flow, avatar wire contract, running locally + remote, full `.env` reference).
> (The parent `E:\Claude\CLAUDE.md` describes a *different* repo, the `.claude` config
> workspace, and does not apply here.)

## What this is

A real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
Multi-turn, streaming end-to-end. Goal: time-to-first-output **< 3 s**. Built on
**Pipecat 1.3.0**, WebRTC to a browser at `/client`.

**Current stack (fully local TTS + avatar). See `STATUS.md` for the full state + the
A/V-sync architecture decision (read it before touching sync).**

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en-US`/`zh-TW`/`th` by `LANGUAGE`) default; **local OFFLINE alt `STT_PROVIDER=sherpa`** (sherpa-onnx streaming zipformer, bilingual zh-en, in-process CPU/~0 VRAM, zh→Traditional via OpenCC) or `funasr` (SenseVoice segmented, `:8004`, untested alt) | cloud / **local CPU** |
| LLM | `LLM_PROVIDER=openrouter` — OpenAI-compatible, so **cloud OR local Ollama** by `OPENROUTER_BASE_URL` (any model via `OPENROUTER_MODEL`); or `weather_chain` (Chinese weather bot) | cloud / local / remote |
| TTS | **CosyVoice2-0.5B** local streaming (default, on vLLM in WSL, TTFB ~1.1s), or **MOSS-TTS-Realtime** (`TTS_PROVIDER=moss`, `:8003`) | **`:8001` cosy / `:8003` moss, both WSL** |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait), **TensorRT render by default** (`MUSETALK_TRT=1`) | **`:8002`, `musetalk` conda env** |
| Config | **Web config panel** — edit `.env` + restart the pipeline from a browser | **`:7870` (`:8444` over Tailscale)** |

**TTS note:** CosyVoice runs its autoregressive LLM on **vLLM inside WSL Ubuntu**
(`cosyvllm` conda env on the Blackwell 5060 Ti) — this cut first-chunk latency ~3.4s→~1.1s, the root
cause of the avatar lip-lag. The pipeline reaches it via `COSYVOICE_URL` set to the **WSL IP**, NOT
`localhost` (WSL2's localhost relay buffers the streaming audio ~2s). Run it with
`bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh` in WSL. The original Windows `tts`-env
PyTorch server is the fallback (set `COSYVOICE_URL=http://localhost:8001` + start it). Full build
notes + gotchas: the `project-visualllm-cosyvoice-vllm` memory.

**Chinese TTS fix (2026-07-02, baked into the cosyvoice repo — `docs/PROBLEMS-AND-FIXES.md` P18):** running
the LLM on vLLM had **dropped CosyVoice's repetition-aware sampling (RAS)**, so zh intermittently looped on
the silence token → a ~4s sentence became ~12s of dead silence (heard as "halting" speech; the avatar kept
moving through the silence). Fixed by **restoring RAS as a vLLM logits processor**
(`CosyVoice/cosyvoice/vllm/ras_logits_processor.py` + `top_p=0.8` in `llm.py`; vLLM's own
`repetition_penalty` CANNOT be used — it CUDA-asserts on the `prompt_embeds` input). Separately, the zh
voice was choppy vs en purely because of the **reference clip** (`zero_shot` clones its rhythm); the baseline
now uses the fluid **"pro" AI-assistant voice** (`CosyVoice/asset/pro_ref.wav`, default in `tts_engine.py`) →
zh ≈ English pacing. An optional zh pause-trimmer (`COSYVOICE_SILENCE_CAP_S`, `_squeeze_silence`) is **OFF by
default** (not needed with the pro voice). Swap voices via `COSYVOICE_PROMPT_WAV`/`COSYVOICE_PROMPT_TEXT`.

**vLLM CUDA graphs — CLOSED: keep EAGER (`COSYVOICE_VLLM_EAGER=1`); graphs degrade zh LIPSYNC (2026-07-05, 8th session, `docs/PROBLEMS-AND-FIXES.md` P27/P31/P32/P33):**
`COSYVOICE_VLLM_EAGER` default is **`1`** (eager) in `run_vllm_server.sh`. P27 set it `0` (CUDA-graph capture) to cut
TTS first-chunk avg ~2.0→~0.85s; P31 reverted it for "live inconsistency." A P32 re-investigation then measured the
TTS side directly (`cosyvoice-local-tts/_ttfb_variance.py`) and found graphs are actually **faster + lower-variance**
than eager — even under real MuseTalk render (96 samp: graphs 1.29/2.23/0.37s vs eager 1.94/3.43/0.64) — so the P31
"shape-spike" mechanism did NOT reproduce. **But that measured the WRONG side.** P33: the real cost is zh **lipsync** —
graphs ON alter the zh AUDIO (measured `cosyvoice-local-tts/_zh_audio_ab.py`: longer + more internal silence, more
variance) because the graph decode perturbs the zh-critical **RAS** sampling (the P18 fix). MuseTalk lip-syncs off a
**Whisper of the waveform**, so a degraded zh waveform → mouth shapes that don't track the words; en is spared (no RAS
reliance), render fps stays ~14 (not a render-starve). **Verdict: eager — graphs win the TTS stopwatch but lose the
avatar, and the avatar is the product** (reconciles with P31's revert; the eye caught what the TTS probe can't see).
The independent Lever-4 poll-tighten (`model.py` 0.1→0.02) stays. Force graphs (only for an en-only / TTS-throughput
setup) = `COSYVOICE_VLLM_EAGER=0` + relaunch, or the config panel's **CUDA graphs** toggle (rewrites the script + relaunches WSL).

**Shared-GPU VRAM (why "won't talk" can mean CosyVoice crashed):** vLLM and MuseTalk share the one
16GB card. vLLM's `gpu_memory_utilization` (env `COSYVOICE_VLLM_GPU_UTIL`, **default `0.3`**, set in
the cosyvoice repo) must exceed vLLM's own ~4GB footprint or load crashes with "No available memory
for the cache blocks" (the old hardcoded `0.2` = 3.26GB was too low). If the avatar shows but the bot
is silent, first check `:8001` is actually up — the pipeline log shows "Cannot connect to host …:8001".
Free VRAM (close a heavy GPU app) or nudge the util fraction; the "Available KV cache memory" log line
must be positive. **LOAD ORDER MATTERS: start CosyVoice (vLLM) BEFORE MuseTalk.** At `gpu_util 0.3` vLLM
needs the card mostly free; if you restart cosyvoice *while MuseTalk already holds ~5GB*, vLLM crashes
"No available memory for the cache blocks" (and raising util then trips "Free memory … less than desired").
Clean recovery = stop all three → start cosyvoice on the near-empty card (`run_vllm_server.sh`) → then
`scripts/run.ps1` (MuseTalk + pipeline). The launcher already does this order. (`docs/PROBLEMS-AND-FIXES.md` P15.)

**Chinese first-chunk is slower than English, and each language has its own TTFO lever (updated
2026-07-04 — the 2026-07-03 hop_zh=5 verdict is REVERSED, see P22).** CosyVoice's first-chunk TTFB
scales with the INPUT sentence length (it prefills the whole sentence before the first audio token). The levers:
- **`COSYVOICE_FIRST_HOP_ZH=0` (baseline since 2026-07-04, default `:-0` in the cosyvoice repo's
  `run_vllm_server.sh`).** hop=5 (a smaller opening TTS chunk) was the 2026-07-03 zh lever, but a live A/B
  proved it HURTS live zh TTFO: the small chunk fills the `MUSETALK_LEAD_FRAMES=14` synced-start cushion
  slowly → the steady-hold balloons (zh hold ~1.9–2.2s vs en ~0.85s; the entire zh-vs-en TTFO gap). hop 5→0
  cut zh median **4.14→3.09s**, screen clean, lips-start *improved*. Its isolated-TTFB win never survives the
  synced-start fill (P19's caveat, now resolved: `docs/PROBLEMS-AND-FIXES.md` P22). English was always hop=0
  (`COSYVOICE_FIRST_HOP_EN`; the old global hop=5 pushed en lip-start ~0.70→~1.95s).
- **`COSYVOICE_FIRST_PIECE` (the first-clause split, `.env`) = the en lever.** en's long sentences benefit
  from starting speech on the first *clause* early. Splits at ASCII comma/space past MIN/MAX char thresholds.
- **`COSYVOICE_FIRST_PIECE_ZH=1` (2026-07-04) = the zh lever** (`docs/PROBLEMS-AND-FIXES.md` P23). The en
  split never fires for zh (ASCII comma/space vs zh's full-width ，and no spaces), so a long zh opener — the
  LLM ignores the ≤10-char-opener prompt rule on ~30% of turns — still prefilled whole (TTS TTFB ~3.1s, turn
  ~4.8s). The zh path flushes the turn's first piece at a full-width **，；： ONLY, never a char cap** (a cap
  cuts mid-word — 天氣預|報 — the rejected splitter; a comma boundary cannot), guarded by
  `COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS=5` CJK-counted chars (the opening piece's audio must cover the next
  piece's synthesis or the voice pauses between clauses). Live A/B: long-opener turns **4.78→3.08s**,
  split-fired audio gaps 59–65ms (no pause); comma-less zh + en byte-identical.
  (`local_services/first_piece_aggregator.py`; knobs read via `os.getenv` inside the aggregator.)

**zh turn-start "breathing sound" — KNOWN/ACCEPTED, no fix in baseline (2026-07-05, `docs/PROBLEMS-AND-FIXES.md`
P34).** CosyVoice's zero-shot synth prepends a low-level breath (25–610ms, −34..−68 dB) before the first word on
~every zh piece; the avatar lip-syncs off a Whisper of the waveform so the mouth moves over it ~0.3–0.6s before the
answer. A start-of-turn byte-stream trim was tried and **REJECTED** (crashed the first piece on aiohttp's odd-sized
chunks → "only speaks one sentence per turn"; user judged no-trim better). The breath is accepted as baseline; any
re-attempt must trim **server-side in CosyVoice** (whole buffers), not in the pipecat client.

`MUSETALK_LEAD_FRAMES` below 14 is a **CLOSED question (2026-07-04): REJECTED by the user's live eye.**
lead=8 measured zh 3.03/en 2.48 median at hop=0 (the first all-under-3s config, probe-screen clean), but the
user live-tested every value below 14 and saw delay or avatar freezes — the probe screen misses what the eye
catches (P19's lesson, twice now). Don't re-try lower leads. The remaining TTFO levers are the TTS first-chunk
cost, the P20 shared-GPU collision (stagger / stream-priority, untried), and the structural fix: a dedicated
avatar GPU. NOTE: lead reaches the avatar server only via a full relaunch (launcher/`run.ps1`) — the config
panel's Restart cycles the pipeline only, so a panel-edited lead never takes effect.

Each stage is a thin single-provider factory in `pipeline/stages/` chosen by `.env` — these
are **deliberate fallback switches, not multi-provider branching**:
- `TTS_PROVIDER` = `cosyvoice` (default) | `moss` (local MOSS-TTS-Realtime, `:8003`) | `elevenlabs` | `deepgram`.
- `LLM_PROVIDER` = `openrouter` (default; point `OPENROUTER_BASE_URL` at `https://openrouter.ai/api/v1` for
  cloud or `http://localhost:11434/v1` for a local Ollama model) | `weather_chain` (NCU zh weather bot).
  **`OPENROUTER_PROVIDER_ONLY` (2026-07-04, TTFO lever):** pin OpenRouter to a fast backend (default
  **`Groq`**) instead of the default transpacific Gemini route — the LLM hop was the dominant TTFO cost +
  all its variance. Injected as `extra_body.provider.only` via pipecat's `Settings.extra` (`stages/llm.py`).
  Cut the LLM hop ~1.1–1.6s (tail to 3.6s) → **~0.7s tight** (zh 1.64→0.80s median, en 1.07→0.67s). Empty =
  unpinned Gemini, fully revertible. End-to-end TTFO only modestly down (TTS + steady-hold now dominate); the
  real prize is the killed 7–8s LLM-tail. **Model baseline = `meta-llama/llama-4-scout`** (Groq, non-reasoning):
  same speed as `llama-3.3-70b`, clean *substantive* Traditional zh, and ~5× cheaper ($0.11/$0.34 vs the 70b's
  real Groq price $0.59/$0.79). Rejected: `llama-3.1-8b` (zh errors), all mid-cost models (reasoning → slower).
  Judge model quality with an ISOLATED probe — `pipeline.log` never logs a turn's reply text. `docs/PROBLEMS-AND-FIXES.md` P21.

**The web config panel (`local_services/config_panel/`, `:7870`) is the easy way to change all of this**
— it edits `.env` in place (preserving comments) and restarts the pipeline. Run it with the system
Python: `python -m local_services.config_panel.server`. Its Restart kills `:7860` via a native Win32
`TerminateProcess` (NOT `taskkill`/PowerShell — those hang for tens of seconds under CPU load here).

**MOSS-TTS-Realtime (`TTS_PROVIDER=moss`):** a streaming server (`local_services/moss_server/app.py`,
`moss-tts` conda env) speaking the SAME `/tts/stream` raw-PCM contract as CosyVoice, so it reuses the
CosyVoice client pointed at `MOSS_URL`. The voice is a fixed reference clip (`MOSS_REF`, clone-only).
**Run it eager** (`TORCHDYNAMO_DISABLE=1`, the default) — compiled mode recompiles ~3–40s on each new
sentence-length, felt as between-sentence stalls. Launch recipe (incl. the `CC`/triton + `torchcodec`
ffmpeg-7/`nvidia-npp`/`LD_LIBRARY_PATH` fixes) is in the server's module docstring.

Core `.env` knobs: `LANGUAGE` (en/zh/th), `TTFO_TARGET_SECONDS`, `TTS_PROVIDER`,
`MUSETALK_SYNC_MODE` (**`steady`** = video-master, synced start, the user's pick and current
default; `live` = audio-master, voice instant + lips trail ~0.75s, can never pause. The old
**steady "screech" is FIXED** — it was pipecat discarding the partial audio buffer after a >3s
render-stall gap (`BOT_VAD_STOP_FALLBACK_SECS`); see `docs/PROBLEMS-AND-FIXES.md` P3 +
`main.py::_relax_bot_vad_stop_timeout` and `musetalk_video.py::_align_even`. Remaining steady
tradeoff: under a long render stall the voice briefly **pauses** then resumes clean — switch to
`live` if that pause is worse than the lip trail), `MUSETALK_FPS` (**14** now (the user's pick); a divisor
of 16000 (8/10/16/20/25) makes frame count = audio length exactly, but the server's `samples_for_frames`
ceil sizing makes a non-divisor like 14 correct anyway; the old `int(16000/fps)` truncation lost ~1 frame/segment → lips
finished ~1–2s early, `docs/PROBLEMS-AND-FIXES.md` P9. NOTE: the end-of-turn leftover-audio blip
(P10) is **FIXED** — `int()`→`math.ceil` on the `audio_cap` in `musetalk_video.py::_advance` so the
final audio sub-frame releases in step instead of waiting for the delayed `video_end` drain),
`MUSETALK_FEED_BURST_S` (1.0 — bursts the first 1s of a turn's audio un-paced so the renderer
isn't starved at turn start; cut lip-start lag ~1.9s→~0.8s), `MUSETALK_END_TAIL_FRAMES` (**0** now —
the client close-crossfade replaces the neutral tail; `>0` = static neutral frames after speech, the
old clean snap), `MUSETALK_CLOSE_FADE_FRAMES` (**5** — eases the mouth shut at end of turn: the client
cross-dissolves the last spoken frame→rest pose over N frames, delivered **free-run/untagged** ("live
during the close") so it survives steady's non-live transport without the audio-cap stranding it; `0`
= clean snap; needs `END_TAIL=0`; `docs/PROBLEMS-AND-FIXES.md` P12), `MUSETALK_IDLE_MOTION` (**0** = no breathing idle; the face
holds the static neutral portrait between turns — the user's pick. `1` = the synthesized breathing
loop. Server reads it from the OS env, so `run.ps1` propagates it), `MUSETALK_SIZE` (**512** now — the delivered
frame px, couples server+client+`video_out`. Bumped from 256 for a crisper studio/hair (the model still generates
the face at a fixed 256px, so higher res only sharpens the STATIC frame, not the animated mouth). **512 is the
lag-free ceiling on this shared GPU: 768/1024 profiled with render headroom in ISOLATION but dropped to ~10fps under
live CosyVoice GPU contention → steady-mode voice lag** (`docs/PROBLEMS-AND-FIXES.md` P36); higher res needs a
dedicated avatar GPU. Also pair with `MUSETALK_BASE_MAX` (source-portrait res cap, **768**; higher = sharper
background but heavier composite) and keep `MUSETALK_FPS` identical across server+pipeline or you get drift),
`MUSETALK_TRT` (**1 = default, load-bearing for A/V sync**: TensorRT UNet+VAE render path,
per-segment render ~389ms→~255ms so the avatar keeps ~12fps under CosyVoice's shared-GPU
contention — where the PyTorch path drifts seconds behind the voice on long turns. Engines live in
`musetalk_server/trt_cache/` (~1.75GB, gitignored, GPU/driver-specific — rebuild with `trt_build.py`);
any load failure silently falls back to PyTorch. `0` = PyTorch. `docs/PROBLEMS-AND-FIXES.md` P16),
`MUSETALK_GPU_COMPOSITE` (**1** — runs the per-frame mask-blend + downscale on the GPU (torch) instead
of CPU PIL/cv2: composite ~73ms→~11ms per 8-frame seg → total render 246→182ms (−26%, ceiling ~33→44fps).
**Only active with `MUSETALK_TRT=1`** (the VAE output is already a GPU tensor there; the PyTorch path
keeps the CPU composite). Output is pixel-identical — SSIM 1.0, ≤1 LSB vs the CPU path. Falls back to CPU
if a crop_box runs off-frame. Code default off (opt-in); `app.py::_composite_gpu`. **Benchmarked: at 12fps
it does NOT reduce A/V drift** (TRT already holds render ≥12fps, even under 100% GPU contention) — the win
is reserve headroom + a freed CPU, judged by the live call. `docs/PROBLEMS-AND-FIXES.md` P17),
`MUSETALK_LEAD_FRAMES` (**14, load-bearing, CLOSED at 14** — it IS the synced-start delay AND a mid-turn
shock absorber; lower starves the queue → freeze. The 2026-07-03 sweep rejected lowering it (P19), and on
2026-07-04 the user live-eye tested **every value below 14** and saw delay or avatar freezes — even `lead=8`
at hop=0, which had measured zh 3.03/en 2.48s median probe-screen clean. The probe misses what the eye
catches; do not re-try lower leads. Server-side knob: only a full relaunch applies it, not the panel Restart),
`COSYVOICE_PACE_RATE` (1.3, in the cosyvoice server — caps voice production so it doesn't burst
the shared GPU), `COSYVOICE_FIRST_PIECE` (**1 = default, TTFO win**: emit a short opening CLAUSE to
TTS first, then normal sentences. CosyVoice's first-chunk TTFB scales with the INPUT sentence length
— it prefills the whole sentence before the first audio token, so a 16-word opener costs ~3.0s vs
~1.7s short. Splitting cut TTS first-chunk ~3.0s→~1.7s and **TTFO ~4.6s→~3.2s**, flow stays smooth
(delivered audio gap ~55ms, never a stall). `COSYVOICE_FIRST_PIECE_MIN_CHARS`/`_MAX_CHARS`
(**18/32** = tuned sweet spot: use an early comma if present, else cap at ~32 chars on a word
boundary — enough opening audio to cover the next piece's synthesis even under shared-GPU
contention; smaller MAX = faster start but risks a between-clause pause). `0` = off.
`local_services/first_piece_aggregator.py`), `COSYVOICE_FIRST_PIECE_ZH` (**1 = the zh split,
2026-07-04**: the en split above never fires on Chinese — full-width ，vs ASCII comma, no spaces —
so long zh openers cost ~3.1s TTFB; this flushes the first piece at a full-width ，；： ONLY, never
a char cap, min `COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS`=5 CJK chars. Long-opener turns 4.78→3.08s, no
between-clause pause. P23), `FILLER_WORDS` (**1 = default baseline, 2026-07-05**: the turn OPENS on a
rotated natural "thinking" phrase ("嗯，讓我想一下喔，…") synthesized through the normal TTS path — one
continuous turn, zero screech (audio gap ~60ms) — so the avatar starts talking + lip-moving ~0.7s sooner
(zh def 2.91→2.23s, wx 2.38→2.03s). **HONEST: a PERCEPTION win, not a speedup — TTFO counts time-to-first-SOUND
and that sound is the filler; the real ANSWER arrives slightly LATER (queued behind it).** **zh CAVEAT (2026-07-05, P30): the filler makes zh feel DELAYED** — the avatar starts on the filler, then the
real zh answer (chopped into short comma/sentence pieces, each ~0.8s TTFB) lags behind, and each short piece's
audio barely covers the next piece's synth → micro-gaps read as "avatar first, voice delayed." en escapes it
(longer fillers + char-count splitting = bigger pieces). Fix = `FILLER_WORDS=0` (confirmed smooth in zh; costs
~0.7s TTFO) or make the filler en-only. NOT the raw TTS speed — zh TTFB ≈ en (~0.9s) once graphs were off.
Fillers are ~1.2s
each so the first piece fills the `MUSETALK_LEAD_FRAMES` cushion — a too-short "嗯，" ballooned the hold 0.5→1.7s
(the fix). `FILLER_WORDS_COUNT` (**1**) chains more for a longer opener. Needs `COSYVOICE_FIRST_PIECE=1` (shares
that aggregator); `0` = off. `local_services/first_piece_aggregator.py`, P26), `CLIENT_FORCE_SPEAKER` (**1 = default**: phone browsers play the voice
on the LOUDSPEAKER, not the earpiece — Android Chrome flips to ear-style 'communication' routing
while the mic is live; iOS gets a WebAudio fallback. Mobile-UA only, desktop/headphones untouched;
the phone self-reports to `[speaker-debug]` in pipeline.log. P24),
`CLIENT_JITTER_BUFFER_MS` (raise only for a remote/WAN viewer),
`WEBRTC_VIDEO_BITRATE_MAX` (caps aiortc's VP8 ceiling so the video fits a WAN link), and
`WEBRTC_ICE_SUBNET` (**`100.64.0.0/10`** = pin WebRTC ICE to the Tailscale interface; fixes the
intermittent remote mic — `0` disables). **Full reference: `WORKFLOW.md` §8.**

## Commands

There is **no build/lint/unit-test suite** — don't invent one. The real commands (3 processes;
`scripts/run.ps1` starts the avatar server + pipeline and propagates the MuseTalk env from `.env`):

**One-click full stack (easiest):** double-click **`Run VisualLLm.exe`** in the repo root. It runs
`scripts/launch.ps1`, which brings up the WSL CosyVoice TTS (waits on `/health`), then `run.ps1`
(avatar + pipeline), then the config panel, then opens `/client/`. The launcher window is the
on/off switch — press Enter (or close it) to stop everything. The `.exe` is a tiny C# shim compiled
from `scripts/Launcher.cs` by the bundled `csc.exe`; rebuild it with `.\scripts\build-exe.ps1` (only
needed if you change `Launcher.cs` — editing `launch.ps1` needs no rebuild). The individual commands
below are still the way to run/debug a single stage.

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

# 3. Pipeline — project main env (SYSTEM Python 3.11, has pipecat — NOT a conda env); serves /client + /nimbus
python -m pipeline.main            # prebuilt: localhost:7860/client/  |  custom Nimbus UI: localhost:7860/nimbus/

# --- or start the avatar server + pipeline together ---
.\scripts\run.ps1

# 4. (optional) MOSS-TTS-Realtime server -- `moss-tts` conda env, serves :8003 (TTS_PROVIDER=moss).
#    Needs CC/triton + torchcodec(ffmpeg7)+nvidia-npp+LD_LIBRARY_PATH; full recipe in the docstring:
#    python -m uvicorn local_services.moss_server.app:app --host 0.0.0.0 --port 8003

# 5. (optional) Web config panel -- SYSTEM python (it restarts the pipeline), serves :7870.
python -m local_services.config_panel.server               # edit .env + restart from the browser

# Verify every fragile import resolves WITHOUT keys/network (Pipecat drift check):
python -m scripts.preflight

# Avatar A/V test tooling (headless, no browser; close any /client tab first — server is single-client):
# UNIFIED harness (PREFER THIS): one command = WebRTC probe + pipeline.log parse + offline capture ->
# output/measure_report.json + docs/measure_data.js (docs/workflow-timeline.html auto-uses it on reload).
python -m scripts.measure --offline-capture                        # full turn timeline + handoffs + metrics
#   measure.py ALSO reports a per-stage LATENCY WATERFALL to the user's EAR (not just server-side [TTFO]):
#   it stitches the same-box probe arrival onto the log's t0 (t0.timestamp() == the probe's time.time()) so
#   the last mile (transport + WebRTC encode + network, then browser jitter/playout) is measured, summing to a
#   true end-to-end. Two last-mile sources: (1) HEADLESS always-on -- the audio pump records (epoch, rms) per
#   frame, first sustained energetic frame after t0 = the answer reaching the client (the `probe` row); (2) REAL
#   BROWSER opt-in `CLIENT_PLAYOUT_PROBE=1` (default OFF) -- a <head>-injected AnalyserNode taps the bot audio
#   and beacons `[client-playout]` first-voice-onset to a new /client/playout endpoint; then:
python -m scripts.measure --from-browser                           # parse-only: fills the `browser` playout row
#   Missing/pre-t0 anchors render `unknown` (never a fake/negative latency); a staleness guard blanks the client
#   arrival if the last [TTFO] turn is older than duration+tail+15s. NOTE: synthetic-mic drives can VAD-split the
#   wav's internal pause -> the LLM row shows `unknown`; a real human turn populates it. `pipeline/metrics.py`
#   (the TtfoMeter) is deliberately UNTOUCHED -- the waterfall is derived in measure.py.
#   (the two tools below are what measure.py wraps; run them standalone only for one-off debugging)
python -m scripts._webrtc_probe --mic output/q_ai.wav --lead 8     # drives a turn, records + metrics
E:\miniconda3\envs\musetalk\python.exe -m local_services.musetalk_server._capture output/q_ai.wav  # offline mp4
# A/V-SYNCED offline capture (keeps ONLY real video_start..video_end frames, auto-detects frame size):
E:\miniconda3\envs\musetalk\python.exe scripts\_capture_synced.py output/q_ai.wav

# FRAMES-vs-AUDIO + DRIFT method (how the P16 numbers were measured; no CosyVoice/pipeline/WebRTC):
# 1) start the MuseTalk server ALONE with the prod env (MUSETALK_TRT/FPS/SIZE from .env), MUSETALK_PROFILE=1
#    for per-8-frame-segment cost (logs feat/whisper/gpu/composite ms -> is render >= the fps budget?).
# 2) drive a WAV and read the THREE distinct counts + effective render fps:
python -m scripts._drive_frames output/reply_concise.wav 12          # paced (default) | burst (pure render)
#    - REAL rendered (server video_clock) = audio_sec*fps (+/-1)  <- lips are never short (P9/P10)
#    - DELIVERED > that by the pump's HELD/duplicate frames (frozen frame kept when render < fps)
#    - drift ~= audio_len * (1 - render_fps/fps); it only SCALES with turn length once render < fps
# 3) reproduce the shared-GPU drift offline (CosyVoice stand-in) — prove MUSETALK_TRT=1 holds >=fps:
E:\miniconda3\envs\musetalk\python.exe scripts\_gpu_contention_hog.py 4096   # run alongside step 2
# GOTCHA: _drive_frames paces the feed with ABSOLUTE deadlines, NOT cumulative asyncio.sleep(0.02) —
# on Windows the ~15ms timer granularity makes a cumulative-sleep feed ~40% slow and FAKES drift.

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
`UserStoppedSpeakingFrame` → `BotStartedSpeakingFrame` (the <3 s metric).

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
  **one** sanctioned mechanism is the `<head>` script injection in `main.py`: env-gated patches
  register into the shared `_client_head_patches` list and ONE middleware serves the index with all
  of them (two separate index-serving middlewares would shadow each other — last-added runs
  outermost and wins). Current patches: the jitter buffer (`_install_client_jitter_buffer`,
  `CLIENT_JITTER_BUFFER_MS`) and the phone speaker route (`_install_client_speaker_route`,
  `CLIENT_FORCE_SPEAKER`, P24 — also handles the `POST /client/speaker-debug` beacon). The index is
  served `Cache-Control: no-store` (a phone that cached the pre-patch page misses every fix). Keep
  new client behavior to that same pattern (env-gated, bundle untouched) rather than forking the
  prebuilt dist.
- **Open the client at `/client/` WITH the trailing slash** — the prebuilt page references its
  assets relatively, so `/client` (no slash) 404s them → white screen.
- **The custom "Nimbus AI" client lives at `/nimbus/`** (figma-to-code redesign, `docs/PROBLEMS-AND-FIXES.md`
  P36/**P37**) — a self-contained vanilla-JS page in `local_services/nimbus_client/` (full-screen weather-anchor avatar +
  glass chat panel), served by `_install_nimbus_client` (StaticFiles, `no-store`). It speaks the SAME SmallWebRTC
  signaling (`POST /api/offer`) as the prebuilt bundle — no build step — so it is **additive**: `/client/` prebuilt
  is untouched and stays the fallback. Its extras are two thin server endpoints (same `_inject_client_patches`
  middleware pattern): **`POST /client/say {text}`** injects a typed turn via `LLMMessagesAppendFrame` into
  `_active_task`, and **`GET /client/transcript?since=N`** serves the conversation for the chat bubbles, fed by a
  READ-ONLY `BaseObserver` on the `PipelineTask` (`_TranscriptStore`; taps bot `LLMTextFrame`s + user
  `TranscriptionFrame`s — no pipeline structural change). Open it WITH the trailing slash, same as `/client/`.
  **Chat behavior (P37):** the user's speech streams into a LIVE bubble word-by-word (STT interims → the store's
  `_partial` slot → `/client/transcript` returns `"partial"` → the client polls at 200ms), then commits as **ONE**
  bubble per turn (segments accumulate, commit at `LLMFullResponseStart`; committing per `TranscriptionFrame` gives a
  bubble per speech pause). The user commit keys on the frame TYPE, **NOT** `frame.finalized` (Deepgram's streaming
  path leaves it False → gating on it dropped every user bubble). The mic button **mutes** (toggles the audio track,
  not disconnect) once connected. **Single-connection:** a new `/api/offer` disconnects the previous session
  (`_active_connection`) so two clients never fight the single-client avatar server.

## Conventions

- Keep stage factories single-provider and thin; config is `.env`-driven only.
- Comments state the *why* (latency, a Pipecat quirk, a hardware constraint) — match that voice.
- Accepted tradeoffs (see `STATUS.md`): echo-guard defaults OFF (`ECHO_GUARD=0`, barge-in — use
  headphones) because the half-duplex mute (`=1`) is broken under the default `steady` sync (mic
  stuck-muted after a turn, `docs/PROBLEMS-AND-FIXES.md` P11); `=1` is valid only with
  `MUSETALK_SYNC_MODE=live`. **`ALLOW_INTERRUPTIONS` (default `1`, live `.env` = `0`, P37):** `0` = the
  bot always finishes its reply (user speech during playback never cancels it). This flips the
  turn-START strategies' `enable_interruptions` (the barge-in broadcast) — NOT the P11-broken mic mute,
  so it is safe under steady. On the single shared GPU the
  lips can trail the voice under load in `live` mode — that's the cost of `live` never freezing; the
  SAFE next lever is bounding the avatar server's `out_q`, **never** re-locking the voice (locked
  sync froze it — see STATUS.md).
- Pipecat import paths drift between releases; the fragile ones are isolated to
  `pipeline/stages/*.py`, `pipeline/main.py`, `pipeline/metrics.py`. Run `python -m scripts.preflight`
  after touching them.
- **Remote viewing** (RDP into this box, or the live avatar over a `tailscale serve` HTTPS URL in a
  remote browser) has its own pitfalls: RDP adds video choppiness AND
  desyncs audio/video — when judging avatar smoothness/sync, use `_capture.py` (offline, no
  WebRTC/RDP) or a native remote browser, never the RDP window; re-encode any mp4 trims
  (`-ss -c copy` breaks playback). When the avatar "won't show" or "won't talk," first check both
  processes are up (`:7860` and `:8002`) and that the pipeline picked up the latest code (restart
  it; a stale process lacks recent fixes).
