# VisualLLm — System Presentation

### Real-time Speech → STT → LLM → TTS → Photoreal Talking-Head Avatar

You speak. The system transcribes you, an LLM answers, the answer is spoken in a cloned voice, and a
GPU renders a **photoreal, lip-synced female face** speaking it back — multi-turn, fully streaming
end-to-end over **WebRTC** to a browser. Everything heavy runs **locally** on one Windows box; only
the LLM and the default STT touch the cloud.

The single acceptance bar is **time-to-first-output (TTFO)** — from the instant you *stop speaking*
to the instant the avatar *starts speaking* — kept **< 3 s** so it feels conversational. Research
target language is **Mandarin (zh-TW)**; English is the easier prototype.

_This deck covers four things: (1) the device you need to run it, (2) how it runs, (3) TTFO in
detail, and (4) every method used to drive TTFO down. Companion docs: `STATUS.md` (source of truth),
`WORKFLOW.md`, `SETUP.md`, `docs/PROBLEMS-AND-FIXES.md` (the P1–P36 bug/fix catalogue)._

---

## 1. Device specification — what you need to run this

Everything heavy is local. The **one hard constraint is a single GPU with ≥16 GB VRAM**, because the
TTS engine and the avatar renderer **share that one card**.

### Hardware

| Requirement | Why |
|---|---|
| **1 NVIDIA GPU, ≥16 GB VRAM** (dev box: **RTX 5060 Ti 16 GB**, Blackwell `sm_120`) | CosyVoice (vLLM) **and** MuseTalk share one card. Below ~16 GB they fight for VRAM → the avatar stalls or TTS fails to load ("No available memory for the cache blocks"). |
| **Windows 11** | Pipeline + MuseTalk avatar + config panel all run on Windows. |
| **WSL2 (Ubuntu)** | Hosts the fast vLLM TTS path. A Windows-only fallback runs without WSL at ~3× the first-audio latency. The Windows NVIDIA driver provides CUDA inside WSL2 — no separate Linux driver. |
| **~20 GB free disk** | CosyVoice2-0.5B (~2 GB), MuseTalk + face weights, TensorRT engine cache (~1.75 GB), optional models. |

### Software environments

The heavy stages are deliberately **separate processes in separate environments** — each needs its
own GPU/Python setup, and isolating them stops one crash from taking down the others.

| Environment | Where | Runs |
|---|---|---|
| **`cosyvllm` conda env** | WSL Ubuntu | CosyVoice2 TTS on vLLM (`:8001`) |
| **`musetalk` conda env** | Windows | MuseTalk avatar GPU server (`:8002`) |
| **system Python 3.11** (has Pipecat — *not* a conda env) | Windows | the pipeline + web client (`:7860`) |
| Miniconda on **both** Windows and WSL | — | provisions the two conda envs |
| _opt._ `moss-tts` conda env | Windows | alternative MOSS TTS (`:8003`) |

### Cloud accounts (both free-tier)

| Service | Stage | Free tier |
|---|---|---|
| **Deepgram** | default STT (`en-US` / `zh-TW` / `th`) | $200 credit |
| **OpenRouter** | LLM (any model; pinned to Groq) | many cheap/free models |

> Both cloud hops are **optional**: `STT_PROVIDER=sherpa` runs STT fully offline on the CPU, and the
> LLM can point at a local Ollama. Without a comparable GPU box the system **cannot run end-to-end**,
> but `python -m scripts.preflight` still validates every import with no GPU or keys.

### The central constraint — one shared 16 GB GPU

**MuseTalk (Windows) and CosyVoice's vLLM (WSL) share ONE 16 GB card.** This single fact drives most
of the system's hard decisions:

- **Load order matters:** start CosyVoice (vLLM) **before** MuseTalk — vLLM needs the card mostly
  free at load; bringing it up while MuseTalk already holds VRAM fails. (P15)
- **Turn-start contention:** at turn start CosyVoice's opening vocoder burst and MuseTalk's first
  render segment collide on the GPU — historically the root of "the mouth moves before the voice" and
  long-turn lip drift. (P16, P20)
- **The recurring verdict:** many latency levers that win in isolation *lose* on the shared GPU. The
  only guaranteed structural cure for the residual ~0.5–1.5 s is a **dedicated avatar GPU** (the
  avatar's working set is only ~4.8 GB). The shared GPU has been attacked from 6+ angles.

---

## 2. Workflow — how it runs

### Three processes + a browser

| Process | Runs in | Port | Job |
|---|---|---|---|
| **CosyVoice TTS** (`run_vllm_server.sh`) | `cosyvllm` conda env in **WSL** (vLLM/GPU) | **8001** | Text → streamed voice audio. Separate repo `E:\Claude\cosyvoice-local-tts` |
| **Avatar server** (`musetalk_server/app.py`) | `musetalk` conda env (GPU) | **8002** | Voice audio → lip-synced RGB frames (MuseTalk) |
| **Pipeline** (`pipeline/main.py`) | system Python 3.11 | **7860** | VAD→STT→LLM→TTS→avatar glue; serves the web client; owns the WebRTC connection |
| _opt._ **Config panel** (`config_panel/server.py`) | system Python 3.11 | **7870** (`8444` over Tailscale) | Edit `.env` + restart the pipeline from a browser |

```
                                         ┌──────────────── CLOUD (US) ─────────────────┐
                                         │  Deepgram STT (nova-2)   ·   OpenRouter LLM  │
                                         │                              → Groq scout    │
                                         └──────────────────────▲──────────────────────┘
                                          wss: mic audio → text  │  https: prompt → tokens
                                                                 │
  ┌───────────────┐    WebRTC (SmallWebRTC)   ┌──────────────────┴──────────────────┐
  │    BROWSER    │  ──── mic (Opus) up ────► │           PIPELINE   (:7860)         │
  │   /client/    │                           │   system Python 3.11  ·  Pipecat     │
  │ (your device) │  ◄── audio + RGB video ── │   VAD → STT → LLM → TTS → avatar      │
  └───────────────┘                           │   glue  ·  WebRTC  ·  serves /client/ │
                                              └──┬───────────────────────────┬───────┘
                     HTTP POST /tts/stream       │                           │   ws ://…:8002/stream
                     text → PCM  ◄── audio ──    ▼                           ▼   PCM up / frames down
                                       ┌──────────────────────┐   ┌───────────────────────────┐
                                       │   CosyVoice2  TTS    │   │      AVATAR  (:8002)       │
                                       │   vLLM · cosyvllm    │   │   MuseTalk · musetalk env  │
                                       │   env · WSL Ubuntu   │   │   TensorRT lip-sync render │
                                       │   (:8001)            │   │   ─► RGB frames + video_*  │
                                       └──────────┬───────────┘   └───────────────┬───────────┘
                                                  │   both are GPU processes on…   │
                                                  └───────────────┬────────────────┘
                                                                  ▼
                                                ╔═══════════════════════════╗
                                                ║  ONE 16 GB GPU (5060 Ti)  ║  ← central constraint (§1)
                                                ╚═══════════════════════════╝
```

- The **browser only ever talks to the pipeline** (`:7860`) over WebRTC.
- The pipeline talks to the **avatar** over a local websocket and to **CosyVoice** over HTTP.
- The **LLM** runs cloud (OpenRouter → Groq), a local Ollama, or the NCU weather chain — no extra
  local process unless you run Ollama.

**One-click launch:** double-click **`Run VisualLLm.exe`** (repo root). It starts the WSL CosyVoice
server (waits on `/health`), then the avatar + pipeline (`run.ps1`), then the config panel, then opens
`/client/`. The launcher window is the on/off switch, and it enforces the critical **load order**
(CosyVoice before MuseTalk).

### One turn, end to end

`pipeline/main.py` assembles a linear Pipecat `Pipeline`; frames stream through it:

```
mic → transport.input() (+ Silero VAD)
    → STT          (Deepgram nova-2, or local sherpa)  : audio → text  ("你好")
    → aggregator.user()                                : builds the user message
    → LLM          (OpenRouter → Groq)                 : streamed, sentence-by-sentence answer
    → TTS          (CosyVoice2 on vLLM, :8001)         : text → voice audio chunks
    → Avatar       (MuseTalkVideoService → :8002)      : voice → lip-synced video
    → TtfoMeter                                        : measures UserStopped → BotStarted
    → transport.output()                               : audio + video → browser
;   aggregator.assistant()                             : records the bot turn into context
```

**The whole thing streams — and that streaming *overlap* is the #1 reason TTFO is small.** The LLM's
*first sentence* reaches TTS before the full answer exists, and TTS's *first audio chunk* reaches the
avatar immediately. Naively chained, four heavy models in series would take **15–30 s** per turn.

### A/V synchronization — `steady` (default) vs `live`

`MUSETALK_SYNC_MODE` picks who is master when the render can't keep up:

- **`steady` (default, video-master):** the voice is buffered and released *paced to the server's
  real rendered frames* (`video_clock` markers), so the voice waits when the render stalls and never
  drifts ahead — a **synced start** (the user's pick). Tradeoff: under a long render stall the voice
  briefly **pauses**, then resumes clean.
- **`live` (audio-master):** voice forwarded immediately so it **can never freeze**; lips best-effort
  (~0.75 s trail under contention). **Rejected by the user** — the voice leading the lips by 1–2 s is
  worse than an occasional pause.

**Load-bearing coupling (`main.py`):** per-frame A/V pinning only works when the transport is
**non-live**, so `video_out_is_live = not config.avatar_sync_with_audio`. And **one fps everywhere is
load-bearing** — the server frame-drop stride, the client release clock, and `main.py
video_out_framerate` must all equal `MUSETALK_FPS` or audio/video drift.

### How the avatar posts frames — the steady pump + sync markers

The avatar server (`musetalk_server/app.py`) never sends frames the moment they finish rendering.
**WebRTC and mobile decoders freeze on bursty input**, so delivery is decoupled from rendering by two
cooperating parts — a GPU render worker that *produces* frames in bursts, and a pump that *emits* them
at a rock-steady rate:

```
  TTS voice (16 kHz PCM)
        │
        ▼
  ┌────────────────────┐   renders in 8-frame segments (UNet+VAE),
  │   RENDER WORKER    │   each ceil-sized to exactly cover its audio;
  │   (GPU thread)     │   BURSTY — fast when free, stalls under contention
  └─────────┬──────────┘
            │  push rendered RGB frames
            ▼
  ┌─────────────────────────────────────┐   smaller cap ⇒ SKIP stale frames
  │   out_q   (bounded queue, ~600)     │   instead of unbounded lip lag
  └─────────┬───────────────────────────┘   (MUSETALK_OUT_Q = the safe lag lever)
            │  drain ONE per tick
            ▼
  ┌────────────────────┐   emits exactly ONE frame every 1/fps tick — never a burst:
  │   PUMP             │      ● fresh rendered frame        (normal)
  │   (steady-fps loop)│      ○ held repeat of last frame   (render hiccup)
  └─────────┬──────────┘      · idle / neutral frame        (between turns)
            │
            │  ws.send_bytes(frame)          binary RGB, every tick (stream never gaps)
            │  + control markers  ───────►   video_start · video_clock{frames:N} · video_end
            ▼                                (N counts ONLY real rendered frames)
  ┌────────────────────┐
  │      BROWSER       │   releases the buffered VOICE paced to video_clock:
  │      /client/      │   render stalls ⇒ N stops advancing ⇒ the voice WAITS with it,
  └────────────────────┘   so audio can never run ahead of the lips  (steady = video-master)
```

Reading the two parts out of the diagram:

1. **The render worker** (GPU thread) buffers the incoming voice and runs UNet+VAE on **fixed 8-frame
   segments** (`SEG_FRAMES`), each ceil-sized to exactly cover its audio (`samples_for_frames`, §4-J).
   Every rendered RGB frame is pushed into a **bounded queue** (`out_q`, default ~600).
2. **The pump** (async loop) drains that queue at a **rock-steady `1/fps` tick**. On every single tick
   it sends **exactly one frame** over the websocket (`ws.send_bytes`) — never zero, never a burst.
   That frame is one of three things:
   - a **freshly rendered** frame (the normal case),
   - a **held repeat** of the last frame (the render momentarily lagged — a hiccup), or
   - an **idle/neutral** frame (between turns: a gentle breathing loop, or the static rest pose).

   Because a frame goes out every tick no matter what, **the video stream is perfectly continuous** —
   a single skipped send *is* a visible freeze, which is exactly the end-of-turn freeze bug that was
   fixed by always emitting.

**Readiness prime.** At turn start the pump does **not** begin draining until `MUSETALK_LEAD_FRAMES`
(14) frames are buffered, holding the last frame meanwhile. That cushion is what absorbs render
hiccups so the synced voice never stutters — and it is also the structural ~0.5 s "avatar readiness
hold" that shows up in the TTFO budget (§3).

**Sync markers ride alongside the binary frames** and are what make `steady` mode possible. The pump
tags the stream with three JSON control messages:

| Marker | When | Meaning |
|---|---|---|
| `video_start` | first **real** rendered frame of a turn drains | the lip-synced segment has begun |
| `video_clock{frames:N}` | every ~2 real frames | N = count of **truly rendered** frames sent so far — *not* held/idle frames |
| `video_end` | turn is really over (`speech_end` **and** queue drained past a grace window) | segment finished |

The client releases the buffered voice **paced to `video_clock`**. So if the render stalls, the real
frame count stops advancing → **the voice waits with it**, and can never run ahead of the lips
(video-master). A brief mid-turn underflow deliberately does *not* emit `video_end` (that would
re-segment and desync the client) — it just holds the last frame until real frames resume.

**The bounded queue is the safe lag lever.** Under GPU contention a *smaller* `MUSETALK_OUT_Q` makes
the render **skip stale frames** rather than let the lips fall arbitrarily far behind the voice. This
is the sanctioned way to bound lag — the rejected alternative (re-locking the voice to the video)
froze the whole stream.

---

## 3. TTFO in detail

**TTFO = time-to-first-output** = from `UserStoppedSpeakingFrame` (you go quiet) to
`BotStartedSpeakingFrame` (the avatar starts speaking). `TtfoMeter` (`pipeline/metrics.py`) logs
`[TTFO]` per turn plus a median/p95 summary on disconnect. The bar is **< 3 s**.

Because the pipeline streams, TTFO is **not** the sum of four stages back-to-back:

> **TTFO ≈ VAD hold + LLM first-token + TTS first-chunk + avatar readiness**

### Where the time goes now

| Component | Cost | Notes |
|---|---|---|
| VAD end-of-turn hold | ~0.5 s | configured silence before a turn "ends" |
| STT | ~0.1–0.3 s | streams *as you speak* — overlaps, ~free |
| LLM first-token | **~0.7–0.9 s** | Groq pin killed the 7–8 s transpacific tail |
| **CosyVoice first-chunk** | **~1.3–2.1 s (dominant, variable)** | scales with the first sentence's length |
| Avatar readiness hold | ~0.45–0.8 s | the structural `lead=14` cushion fill (not GPU-starved) |
| **End-to-end TTFO** | **en ~2.0–2.5 s · zh ~2.2–3.1 s** | long-opener zh turns still tail over 3 s |

**The dominant, most variable cost is the TTS first-chunk:** CosyVoice prefills the *whole first
sentence* before emitting the first audio token. That is why so many methods in §4 attack it from
different angles.

### TTFO stops at the pipeline — the *true* latency goes to the ear

`[TTFO]` measures to `BotStartedSpeakingFrame` (the pipeline starts pushing audio). But the user only
*hears* the voice after the last mile: transport pacing + WebRTC encode + network + browser jitter
buffer + playout. The measurement harness (`scripts/measure.py`) now decomposes a turn into a
**per-stage waterfall that sums to a true end-to-end number**, by stitching the same-box clocks
(`t0.timestamp()` == the headless probe's `time.time()` == the browser's `Date.now()`).

**A real measured turn — `scripts/measure.py`, 2026-07-06 15:09, this box (RTX 5060 Ti, Blackwell),
`LANGUAGE=zh`.** Question: *"什麼是人工智慧？請用一句話簡短回答。"* Answer: *"就是電腦模擬人腦思考。
您想了解更多嗎？"* Every timestamp below is a real anchor pulled from `pipeline.log` (`t` = seconds
after t0):

| t (s from t0) | Event | Δ | What it is |
|---|---|---|---|
| −2.0 → 0.000 | **User speaking** — Deepgram streams the mic live | — | STT overlaps the whole utterance; partials refine into one final |
| **0.000** | **t0** — VAD says the user STOPPED; STT emits the final transcript → LLM | — | the `< 3 s` TTFO stopwatch starts here |
| 0.000 | LLM receives the transcript (pre-warmed on connect) | +0.00 | no cold start |
| **1.066** | LLM emits first token | **+1.066** | OpenRouter TTFB — the transpacific cloud hop (unpinned this run; the Groq pin cuts this to ~0.8 s, §4-B) |
| 1.078 | TTS receives sentence 1 (*"就是電腦模擬人腦思考。"*) | +0.012 | first complete sentence flushed early |
| **1.953** | TTS emits sentence-1 first chunk (CosyVoice TTFB 0.874 s) | **+0.875** | the single biggest slice of TTFO |
| 1.953 | MuseTalk receives the voice (forwarded real-time-paced) | +0.00 | — |
| **2.898** | **Bot started speaking → `[TTFO]` = 2.9 s** ✅ (target 3 s) | **+0.945** | avatar readiness / steady lead-hold |
| ~4.89 | the (remote) browser actually **plays** the voice | **+1.99** | last mile — remote Tailscale, network-dominated |

Sentence 2 (*"您想了解更多嗎？"*) is synthesized **in parallel** (1.96 → 2.83 s, its own TTFB 0.868 s)
while sentence 1 is already playing, so it costs TTFO nothing — the payoff of sentence streaming (§4-A).

**Collapsed to a waterfall (from t0 to the ear):**

| Stage | Δ | cum | source |
|---|---|---|---|
| STT finalize → LLM | +0.00 s | 0.00 s | assumed (pre-warmed) |
| **LLM first token** | **+1.066 s** | 1.07 s | log |
| LLM → TTS (sentence-1 flush) | +0.012 s | 1.08 s | log |
| **TTS synth first chunk** | **+0.875 s** | 1.95 s | log |
| TTS → bot-start (steady lead-hold) | +0.945 s | **2.90 s = `[TTFO]`** | log |
| Transport + encode + network | *(unknown — no headless probe this run)* | — | probe |
| **Browser jitter + net + playout** | **+1.99 s** | **4.89 s** | browser+net |
| **END-TO-END, user hears** | | **4.89 s** | |

**Headline: `[TTFO]` = 2.9 s (passes), but this remote viewer's ear is reached at ~4.9 s.** The reads:
- **Inside the pipeline, the two dominant costs are exactly the two §4 attacks:** the LLM first token
  (**1.07 s** — this run ran *unpinned*, which is why TTFO is 2.9 s and not ~2.2 s; the Groq pin, §4-B,
  is what removes it) and the CosyVoice first chunk (**0.875 s**). The steady lead-hold adds 0.945 s.
- **The +1.99 s last mile here is a *remote* number.** This turn was viewed over Tailscale, where the
  last mile is network/jitter-buffer dominated and **swings with the link** (measured 0.94 s on one
  run, 1.99 s on another). The pipeline is not the lever there — `CLIENT_JITTER_BUFFER_MS` and the link
  are. A **local** viewer's last mile is far smaller (~0.15–0.8 s, mostly WebRTC/Opus encode plus the
  onset detector waiting out CosyVoice's sub-threshold **zh leading breath**, P34).
- With no headless probe on this run, the transport row is folded into `browser+net`, hence "unknown"
  transport and the `browser+net` label.

**Measurement discipline (twice-proven law):** A/B on the **LIVE full stack** only, and **the user's
live eye is the final gate** — the offline probe has passed multiple configs the eye then rejected.

---

## 4. What we do to minimize TTFO

Methods grouped by what they attack. Each notes whether it **shipped ✅**, is a **perception ⚠️** win,
or was an honest **dead-end ❌**. `Px` = the entry in `docs/PROBLEMS-AND-FIXES.md`.

### A. Architectural overlap (the foundation)
1. **Sentence streaming.** The first sentence reaches TTS before the full answer exists; TTS's first
   chunk reaches the avatar immediately. Without it the pipeline is 15–30 s.
2. **LLM connection pre-warm.** On connect the pipeline opens the TLS connection and speaks a fixed
   greeting with **no LLM round-trip**, so the first real turn pays no cold-start tax.
3. **VAD `stop_secs` tuning.** The end-of-turn silence hold adds directly to TTFO — tuned snappy
   without clipping the user.

### B. Killing the LLM cloud hop — Groq pin (P21) ✅
The LLM hop was the single largest TTFO component **and its entire variance** (a transpacific Gemini
route with 7–8 s tail outliers). Fix: pin OpenRouter to a fast backend via
`OPENROUTER_PROVIDER_ONLY=Groq`. **LLM hop halved, tail killed:** zh 1.64 → 0.80 s median (max 3.59 →
1.44), en 1.07 → 0.67 s. Baseline model **`meta-llama/llama-4-scout`** — same speed as llama-3.3-70b,
clean Traditional zh, ~5× cheaper. A **short-first-sentence prompt** further trims the zh TTS
first-chunk (~0.3–0.5 s).

### C. The original TTS latency fix — CosyVoice on vLLM in WSL (P6) ✅
CosyVoice's autoregressive prefill delivered the opening ~1.2 s of speech over ~1.5–3 s — the real
lip-lag root cause. Fix: run CosyVoice2's LLM on **vLLM in WSL Ubuntu**. Measured **TTFB 3.4 s →
~1.1 s** and it now genuinely streams.

### D. Shrinking the TTS first-chunk by *input* — first-piece / first-clause splits ✅
CosyVoice's first-chunk TTFB **scales with the input sentence length**. So emit a short *opening
clause* to TTS first, then normal sentences (`first_piece_aggregator.py`):
- **English — `COSYVOICE_FIRST_PIECE=1`** (MIN=18/MAX=32). Splits at an ASCII comma/space past the
  thresholds. **TTS first-chunk ~3.0 → ~1.7 s, TTFO ~4.6 → ~3.2 s**, gap ~55 ms.
- **Chinese — `COSYVOICE_FIRST_PIECE_ZH=1`** (P23). The en split never fires on zh (full-width ，, no
  spaces). Flushes the first piece at a full-width **，；：ONLY, never a char cap** (a cap cuts mid-word
  — 天氣預|報 — a comma boundary cannot), guarded by `_ZH_MIN_CHARS=5`. **Long-opener turns 4.78 →
  3.08 s**, gaps 59–65 ms.

### E. The zh quality fixes that unblocked speed — RAS restored + "pro" voice (P18) ✅
Running on vLLM had dropped CosyVoice's **Repetition-Aware Sampling (RAS)**, so zh intermittently
**looped on the silence token** (~40 % of runs) — a ~4 s sentence became ~12 s with ~5 s of dead
silence (the avatar kept moving through it). Fix: reinstate RAS as a vLLM logits processor
(`ras_logits_processor.py` + `top_p=0.8`). Separately, zh choppiness was purely the **reference clip**
→ swapped to a fluid **"pro" voice** (`pro_ref.wav`) → zh ≈ English pacing.

### F. Avatar render — TensorRT (P16) ✅ merged to `main`
Long-turn lip drift was **shared-GPU contention**, not frame math. Fix: **TensorRT** UNet+VAE render
(`MUSETALK_TRT=1`, default): per-segment 389 → ~255 ms, holding ~12 fps under contention. **Under the
same 100 % contention that drifted PyTorch +3.94 s on a 13.6 s reply, TRT holds drift flat at +0.36 s
at every length.** Engines are GPU/driver-specific (~1.75 GB, gitignored); any load failure falls back
to PyTorch.

### G. Avatar render — GPU composite (P17) ✅ opt-in
`MUSETALK_GPU_COMPOSITE=1` moves the per-frame mask-blend + downscale onto the GPU (the VAE output is
already a GPU tensor): composite ~73 → ~11 ms/seg → total render 246 → 182 ms (−26 %),
**pixel-identical** (SSIM 1.0). At 12 fps it does *not* reduce drift — the win is **reserve headroom +
a freed CPU** for STT/pipecat/LLM-streaming.

### H. Avatar turn-start — cudnn.benchmark = False (P1) ✅
With `cudnn.benchmark=True`, cuDNN re-autotuned on the turn-START segment (a different shape than
mid-turn) → a **~16 s GPU spike on the first segment of *every* turn** → lips ~5 s late. `False`
removed it entirely (steady-state per-frame time unchanged). A hard rule in the avatar server.

### I. Avatar turn-start — feed burst (P2) ✅
Real-time-paced feeding starves the renderer at turn start. `MUSETALK_FEED_BURST_S=1.0` bursts the
**first 1 s** of each turn's audio un-paced (then resumes pacing). Cut lip-start lag **~1.9 s →
~0.8 s**.

### J. Frame math — ceil-sized segments (P9/P10) ✅
`samples_for_frames(n) = ceil(n·16000/fps)` sizes each render segment to the frame's upper sample
boundary, so the renderer yields exactly `SEG_FRAMES` per batch for **any** fps. Fixes "lips finish
~1–2 s before the voice," plus a companion `ceil` on the audio-cap that removes the end-of-turn blip.

### K. Removing a cloud hop entirely — local sherpa STT (P29) ✅ opt-in
`STT_PROVIDER=sherpa` runs STT fully offline on the CPU (~0 VRAM), removing the Deepgram round-trip.
Two fixes landed with it: the TTFO meter now also arms on sherpa's `VADUserStoppedSpeakingFrame`, and
sherpa's bot-speech pause no longer strands the mic under `steady`.

### L. Perception — filler-word opener (P26) ⚠️ baseline, a *perception* win
`FILLER_WORDS=1`: the turn **opens on a rotated natural "thinking" phrase** ("嗯，讓我想一下喔，…")
synthesized through the normal TTS path (one continuous turn, gap ~60 ms), so the avatar starts talking
**~0.7 s sooner** (zh 2.91 → 2.23 s). **Honest framing:** TTFO counts time-to-first-*sound*, and that
sound is now the filler — the metric improves but the real **answer arrives slightly later**. A
responsiveness win, not a real speedup. **zh caveat (P30):** the filler colliding with zh's short-piece
splitting can read as "avatar first, voice delayed" — `FILLER_WORDS=0` (or filler en-only) fixes it.

### Honest dead-ends & rejected levers (recorded so they're never re-tried)

A defining discipline of this project: **record what did *not* work, and why, with the same rigor as
the wins.** Every one was measured, not guessed.

| Lever | Verdict | Why (measured) |
|---|---|---|
| **vLLM CUDA graphs** (P27/P33) | ❌ reverted to eager | Faster on the TTS stopwatch (96 samp: 1.29 vs 1.94 s) **but degrades zh lipsync** — the graph decode perturbs the zh-critical RAS sampling → the mouth stops tracking the phonemes (en spared). Keep `COSYVOICE_VLLM_EAGER=1`. *The probe passes what the eye rejects (3×).* |
| **`COSYVOICE_FIRST_HOP` (small opening chunk)** (P19/P22) | ❌ reverted to 0 | Isolated TTFB win **erased live** — the small chunk fills the `lead=14` cushion slowly → the steady-hold balloons. |
| **`MUSETALK_LEAD_FRAMES` < 14** (P19/P22) | ❌ closed at 14 | `lead=8` measured the first all-under-3 s config **but the user's live eye rejected every value below 14** — delay or avatar freezes. The probe misses what the eye catches (twice-proven). |
| **FP8 quantization** of the render UNet (P20) | ❌ dead end | Quality perfect (SSIM 0.99997) but **4.5× SLOWER** — TRT 10.13 has no FP8-conv kernels for Blackwell `sm_120`. Retry when a newer TRT ships them. |
| **Flow-matching TensorRT** (P28) | ❌ rejected | Builds/runs but **zero TTFB win + 26–40 % slower** — the flow isn't the first-chunk bottleneck; TRT's per-chunk hard-syncs swamp the fusion. |
| **GPU stream priority** (`MUSETALK_HP_STREAM`, P25) | ❌ rejected | Catastrophic live (TTFO ~12 s) — cross-process CUDA priority needs Linux MPS; Windows **WDDM thrashes** across the WSL/Windows boundary. |
| **CosyVoice turn-start throttle** (`COSYVOICE_PACE_RATE`, P25) | ❌ rejected | Useless — the TRT render is **not** GPU-starved at turn start; both P20 levers chased a non-bottleneck. |
| **`MUSETALK_SYNC_MODE=live`** (P7) | ❌ rejected by user | Voice leads the lips ~1–2 s — worse than steady's occasional pause. |
| **zh leading-breath trim** (P34) | ❌ reverted | Offline-clean but **crashed live** (`np.frombuffer` on aiohttp's odd chunks). Any re-attempt must trim **server-side in CosyVoice**, not the byte-stream client. |
| **Local CPU-pinned Ollama LLM** | ❌ reverted to cloud | CPU contention with the memory harness made end-to-end latency **worse** + fragmented zh. |

**The through-line:** on a single shared GPU, most latency levers that win in isolation lose in the
live system, and the human eye is the final gate the objective probes repeatedly fail. The remaining
levers, honestly:
1. Further shrink the TTS first-chunk (the dominant, variable cost) — input-splitting is near its
   useful floor; the compute side is capped by the zh-lipsync/RAS constraint.
2. **A dedicated avatar GPU** — the only *guaranteed* structural cure for the residual shared-GPU cost
   (it helps long-reply drift; note it would *not* cut turn-start TTFO, which is lead-fill, not
   contention).
