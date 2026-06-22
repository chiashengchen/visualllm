# VisualLLm — Problems Found & How We Fixed Them

A running catalogue of the real bugs hit in this system, each with the **symptom**, the
**root cause** (how it was actually proven, not guessed), and the **fix**. Newest first.
This is the "why is the code the way it is" reference — read it before re-litigating a fix.

> Companion docs: `STATUS.md` (current state, source of truth) · `WORKFLOW.md` (end-to-end
> workflow + full `.env`) · `CLAUDE.md` (conventions). Live measurement: `scripts/measure.py`.

---

## How we debug audio here (hard-won method)

- **Per-chunk RMS/flatness is unreliable.** Audio chunks (aiohttp `iter_chunked`, pipecat
  frames) are **not sample-aligned** — a chunk can split mid-int16-sample, so analysing one
  chunk alone reads as "loud garbage" even when the concatenated stream is clean. This sent us
  on a multi-hour wild goose chase. **Always judge audio from a CONCATENATED WAV**, windowed
  (≥0.5 s), skipping silence windows (`rms<0.005`), using spectral flatness (noise ≈ 0.5+,
  speech ≈ 0.0–0.05).
- **Bisect at component boundaries with WAV captures**, not vibes: source (CosyVoice) →
  avatar-out → transport-in → transport-out. Whichever boundary flips clean→garbled is the culprit.
- **Reproduce headlessly**: `python -m scripts._webrtc_probe --mic output/q_ai.wav --lead 8
  --duration 42` drives a real turn with no browser. `python -m scripts.measure --offline-capture`
  is the one-command wrapper.
- **The GPU is shared** (CosyVoice TTS + MuseTalk render on one card) — contention is real, but
  GPU math does not *corrupt* under load, it only *slows*. Garbage bytes ⇒ a logic/library bug,
  not contention.

---

## P1 — Avatar starts ~2s late; on long replies "audio ends then the avatar keeps moving ~10s" ✅ FIXED

**Symptom.** First lip frame appeared ~2–5 s after the voice; on a long multi-sentence reply the
render fell progressively behind and the mouth kept moving for several seconds *after* the voice
finished. Felt worst right after a server (re)start.

**Investigation.** Per-stage profiling in the MuseTalk server (`MUSETALK_PROFILE=1`) showed the
**first render segment of every turn took ~16 s on the GPU; every later segment ~240 ms**:
```
[profile] whisper=853ms gpu=16372ms -> 7 frames    # FIRST segment of the turn
[profile] whisper=7ms   gpu=240ms   -> 7 frames    # every later segment (normal)
```
A warmup at load did NOT fix it (it recurred every turn), so it was not a one-time cold start.

**Root cause.** `local_services/musetalk_server/app.py` set `torch.backends.cudnn.benchmark = True`
on the assumption of "fixed input shapes". But the **turn-start segment has a different tensor
shape** than steady mid-turn segments (padding at speech start), so cuDNN re-ran its **algorithm
autotune** on the first segment of *every* turn — and on this shared Blackwell card that
autotune + workspace alloc ballooned to ~16 s.

**Fix.** `torch.backends.cudnn.benchmark = False` (`app.py`). Steady-state per-frame time was
unchanged (240 ms), so benchmark mode was pure downside here.

**Verified.** first-segment GPU **16,372 ms → 346 ms**; lips-start **+5.3 s → +1.0 s**; a 13 s
reply renders 179 frames at a steady 12 fps; server warmup **17.7 s → 1.0 s**.

---

## P2 — Residual ~1.9s lip-start lag (after P1) ✅ FIXED

**Symptom.** Even with P1 fixed, lips consistently started ~1.9 s after the voice (worst on short
replies — the reply was nearly over before the lips moved). Steady ~0.9 s offset then held flat
through the turn (so it was a *startup* lag, not accumulating drift).

**Root cause.** The avatar client (`musetalk_video.py`) feeds audio to the render server
**real-time-paced** (`_feed_q`) so CosyVoice's faster-than-real-time output can't build a video
backlog. But that pacing also **starves the renderer at the very start** of a turn — it can't fill
its lead-frame prime + first segment until ~1 s of audio has trickled in at real-time.

**Fix.** Burst the **first `MUSETALK_FEED_BURST_S` (=1.0 s)** of each turn's audio to the server
*un-paced*, then resume real-time pacing (keeps the no-backlog guarantee). `musetalk_video.py`
`_feed_loop` + `_burst_remaining`, reset on `speech_start`.

**Verified.** lip-start **~1.9 s → ~0.75–1.0 s** across short and long replies; render still 12 fps;
end drift small/negative (no backlog). Instrumentation added: a per-turn **`[avatar timing]`** log
(`lips start +Xs after voice | audio Ys video Zs (N frames, F fps) | end drift +Ds`).

> Note: `MUSETALK_LEAD_FRAMES=14` is **load-bearing** — it's the server's readiness cushion;
> lowering it (tried 6) starves the queue → underflow → **freeze**. Don't lower it to chase the
> last ~0.5 s of startup lag.

---

## P3 — `MUSETALK_SYNC_MODE=steady` intermittently SCREECHES the voice mid-reply ✅ FIXED (2026-06-22)

**Symptom.** In `steady` mode (video-master — voice held and released locked to rendered frames,
which gives a perfectly synced *start* the user liked), ~⅓ of multi-sentence replies the voice
turned into **loud broadband noise from the middle of the speech to the end of the turn**. `live`
mode never did this.

**The earlier (WRONG) conclusion — recorded so it isn't re-derived.** A first pass bisected with
concatenated WAVs and reported "transport INPUT clean, transport OUTPUT garbled, only non-live →
unfixable bug inside pipecat's non-live write path." **That boundary table was wrong** (it relied on
runs that didn't time-align, plus per-frame RMS false positives from sample-misalignment). The
two "rejected fixes" it spawned — a CosyVoice `bytes(chunk)` yield-copy and a MuseTalk
`copy.copy(frame)`/`frame.audio=bytes(...)` snapshot — were aimed at a *frame-aliasing* theory that
isn't the cause. **Both have been removed** (the real fix is `_align_even` below); do NOT re-add them.
The misleading per-chunk-RMS debug logs (`[ms-push h]`, `[pf-in]`, `[cb-yield]`) that produced the
false positives were also removed.

**Investigation that actually nailed it (byte-level, on the live captures).**
1. The screech is a **1-byte (odd) sample misalignment, not generated noise.** Byte-shift sweep of
   the garbled region: offset 0/2/4 → flatness ≈0.52 (noise); offset **1**/3/5 → ≈0.079 (clean
   speech). The speech is fully intact, just read across the wrong int16 boundary.
2. Byte-diffing the clean pre-transport stream (`MuseTalk` push, `MUSETALK_DOWNSTREAM_CAPTURE`)
   against the garbled delivered stream (`COSYVOICE_DELIVERED_CAPTURE`): identical up to **6.040 s**,
   then `delivered[k:] == musetalk_out[k+1049:]` — i.e. **1049 bytes (odd) were DELETED** at one
   point mid-turn; everything after is bit-identical but odd-shifted → broadband noise to turn end.
3. A dropped *partial* buffer points at pipecat's `_bot_stopped_speaking()`, which does
   `self._audio_buffer = bytearray()`.

**Root cause (proven).** pipecat's output transport (`MediaSender._next_frame`) fires
`_bot_stopped_speaking()` if **no audio frame reaches its queue within
`BOT_VAD_STOP_FALLBACK_SECS` (default 3 s)**, and that handler **discards the partially-buffered
audio**. In `steady`/non-live sync the voice is released **paced to rendered video frames**, so a
**> 3 s render stall** on a long reply (shared GPU) starves that queue → the 3 s timeout fires
**mid-turn** → the partial `_audio_buffer` (an arbitrary, usually **odd** byte count, e.g. 1049) is
thrown away → the remaining PCM is left odd-misaligned → screech. `live` never hits it: it forwards
audio **continuously**, so the queue is never starved 3 s. This explains *steady-only*, *long-turn
only*, *mid-reply onward*, and *intermittent (~⅓)* — all of it.

**`_bot_stopped_speaking` (the discard) fires from TWO triggers** — important, the first fix alone
was insufficient:
1. the **> 3 s audio-gap timeout** (`BOT_VAD_STOP_FALLBACK_SECS`) above, and
2. the **per-turn `TTSStoppedFrame`** (`_handle_frame`). In `steady`, MuseTalk pushes
   `TTSStoppedFrame` downstream *immediately* but releases the buffered voice *later* (paced to
   video), so on a long reply the stop signal lands **mid-drain** → `_bot_stopped_speaking` clears
   a still-partial buffer. The odd remainder comes from CosyVoice's **odd-length final chunk** per
   utterance (`iter_chunked`).

**Fix — two layers (defense in depth):**
1. `pipeline/main.py::_relax_bot_vad_stop_timeout()` raises
   `pipecat.transports.base_output.BOT_VAD_STOP_FALLBACK_SECS` (knob `BOT_VAD_STOP_FALLBACK_SECS`,
   default **600 s**, read as a module global at `_next_frame()` call time, patched before connect).
   We already drive an explicit `TTSStoppedFrame`, so the 3 s gap fallback is redundant — a render
   stall can no longer discard the buffer; the voice just **pauses** then resumes clean.
2. **THE root-level fix — `local_services/musetalk_video.py::_align_even()`** (the sample-alignment
   guard). It makes **every** audio frame pushed downstream a whole-sample (**even**) byte count,
   carrying any dangling odd byte to the next frame so the PCM stays exactly contiguous. The
   transport's running total is then always even, so **any** buffer-clear — from *either* trigger,
   or any future one — can only ever drop an even (whole-sample) gap = at worst an inaudible click,
   **never** the half-sample screech. This is what makes the screech impossible by construction.
   (`steady` is now the default in `.env`.)

**Verified.**
- Deterministic regression test `scripts/_screech_repro_test.py`: (a) short timeout + mid-turn gap →
  partial buffer discarded (bug), raised timeout → kept (fix 1); (b) the odd CosyVoice-style stream
  leaves a max **1415-byte ODD** transport remainder without the guard (screech possible), and the
  `_align_even` carry makes the remainder **always even** (fix 2). **PASS.**
- Live (steady, both fixes active): driven turns deliver audio **byte-identical** pre- vs
  post-transport (0-byte diff, 0 noisy 0.5 s windows), vs the pre-fix 1920-/1049-byte deletion.
- Note: short-reply live turns can't *trigger* the stall path, so the guard (proven by construction +
  the deterministic test) is what guarantees the long-turn case, not the short live runs.

**Method note (still true):** judge garble from a **concatenated WAV** windowed ≥0.5 s with spectral
flatness; per-frame/per-chunk RMS lies (chunks aren't sample-aligned) — it sent the first pass down
the wrong path.

---

## P4 — "Works sometimes, mostly not" remote mic = WebRTC ICE candidate pollution ✅ FIXED

**Symptom.** Over a remote Tailscale browser, most sessions ended with zero transcripts,
`TTFO {'count': 0}`, and `Media stream error; clearing track`.

**Root cause.** The box advertised **9 ICE host candidates** (Tailscale, Hyper-V, Radmin, LAN,
APIPA). ICE checked a dead matrix, nominated a marginal pair, then dropped the audio track. Not
bandwidth/TURN.

**Fix.** `main.py::_restrict_ice_to_subnet()` keeps only `WEBRTC_ICE_SUBNET` (default
`100.64.0.0/10`, Tailscale's range) → 9 candidates collapse to 1 working one.

---

## P5 — DEBUG log flood choking the realtime loop ✅ FIXED

**Symptom.** ~41k log lines / 10 MB per 20 min; aggravated P4.

**Root cause.** `log_setup.py` defaulted to DEBUG with the stdlib root at level 0 → aiortc logged
every RTP packet through a stack-walking handler **on the asyncio media loop**.

**Fix.** Intercept root → INFO, aiortc/aioice → WARNING. Verified 41,492 → 0 DEBUG lines.

---

## P6 — Avatar lip-lag ~1.5–2s = CosyVoice first-chunk latency (the original TTS-latency fix) ✅ FIXED

**Symptom.** Lips trailed the voice ~1.5–2 s.

**Root cause.** NOT the render (server starts lips in ~0.77 s if fed fast) but **CosyVoice's
first-chunk latency** — the autoregressive LLM prefill+gen delivered the opening ~1.2 s of speech
over ~1.5–3 s (~3 s fixed per-reply cost).

**Fix.** Moved CosyVoice2's LLM onto **vLLM in WSL Ubuntu** (Blackwell 5060 Ti). Measured TTFB
**3.4 s → ~1.1 s**, and it now actually streams. Pipeline reaches it at the **WSL IP** (NOT
`localhost` — WSL2's localhost relay buffers the stream ~2 s). Full build notes:
`project-visualllm-cosyvoice-vllm` memory + `E:\Claude\cosyvoice-local-tts\run_vllm_server.sh`.

---

## P7 — Locked/steady A/V sync froze the whole voice ✅ AVOIDED (design decision)

**Symptom.** Earlier video-master experiments: any render stall **froze the voice too**
(`hold=2.88s audio 2.9s video 0.0s`).

**Root cause.** On a single shared GPU the render can dip below real-time; locking the voice to the
video propagates a render stall into a voice freeze.

**Decision.** Default to **`live` (audio-master)** — voice forwarded immediately, can never freeze;
lips best-effort. (Steady was revisited this session for its synced start, but see **P3** — it has
the separate pipecat screech bug, so `live` remains the default.)

---

## P8 — TTS dead (avatar shows but won't talk) = ElevenLabs out of credits ✅ FIXED (fallback)

**Symptom.** Replies died at the TTS stage; logged 33 s TTFB was the websocket hanging on a quota
error.

**Fix.** `TTS_PROVIDER` switch in `pipeline/stages/tts.py` — `deepgram` builds Deepgram Aura
(English-only) as a fallback; `cosyvoice` (default) and `elevenlabs` remain. A deliberate single
fallback switch, not a return to multi-provider branching.

---

## Earlier Ditto-path fixes (fallback engine, `AVATAR=ditto`)

The Ditto full-face TensorRT path accumulated its own long bug list — onnxruntime silently on CPU
(missing CUDA DLLs), reconnect freeze (session lock), frame-rate mismatch drift, bursty-render
jerkiness, neutral-snap between sentences, the diffusion-warmup "delay", and the
**`sync_with_audio` no-op under a live transport** (the same `video_out_is_live` coupling that
underlies P3). Those are documented in detail in `STATUS.md` (the dated Ditto sections) and remain
valid for the `AVATAR=ditto` fallback. MuseTalk is the default and the focus of P1–P3 above.
