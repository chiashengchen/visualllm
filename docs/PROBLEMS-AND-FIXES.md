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

## P9 — On long replies the avatar (lips) finishes ~1–2 s BEFORE the voice ✅ FIXED (2026-06-23)

**Symptom.** On longer answers the mouth stopped moving / settled to neutral while the voice kept
talking for the last ~1–2 s. Short replies looked fine. The gap scaled with reply length. Only
appeared after `MUSETALK_FPS` was lowered to 12.

**Investigation.** Matched the server's per-turn `[stream] turn rendered N` against the client's
audio for the same turn:

| Audio (TTS) | Lip frames rendered | Needed @12fps | Lips short by |
|---|---|---|---|
| 13.48 s | **141** | 161 | ~1.7 s |
| 12.52 s | **132** | 150 | ~1.5 s |

The server's own warmup log gave it away: `MuseTalk warmup done … (7 frames/segment)` — `SEG_FRAMES=8`
but each segment rendered **7**. The client's `[avatar timing] end drift` is *misleading* here (it
counts held + `END_TAIL` neutral frames as "video", so it under-reports the gap); the reliable
signal is the server's `turn rendered N` vs `audio_sec × fps`.

**Root cause.** The server sized each render segment as `int(16000/fps) · SEG_FRAMES`, but the
renderer counts frames as `floor(len/sr · fps)` (`audio_processor.get_whisper_chunk`). `int(16000/12)`
truncates `1333.33 → 1333`, so an 8-frame batch measures `floor(7.998) = 7` — **one lip frame lost
per segment**, accumulating ~12.5% over a turn. It only bites at fps that don't divide 16000 evenly,
so **lowering `MUSETALK_FPS` to 12 introduced it** (at the old default 20, `16000/20 = 800` exact → no
loss). The `speech_end` tail-pad (`(-len) % spf`) had the same truncation on the final syllable.

**Fix.** `MuseTalkEngine.samples_for_frames(n) = ceil(n · 16000 / fps)` (`app.py`) — size each segment
to the frame's *upper* sample boundary so `floor()` lands on exactly `SEG_FRAMES` for any fps. Wired
into all four audio→frame sizing sites: stream init, the `config` handler, `_warmup`, and the
`speech_end` tail-pad. Frame count is now `= audio_sec × fps` end-to-end. **`MUSETALK_FPS` should
divide 16000** (8/10/12.5/16/20/25…); the fix makes 12 correct anyway.

**Verified (live, fps=12).** Warmup `7 → 8 frames/segment`; a 13.56 s reply rendered **163** lip
frames (was 141) = its full audio length, so the lips end with the voice. Deterministic repro (no
GPU): `archive/_frame_deficit_repro_test.py` (old=142 ≈ the observed 141, new=162). Live driver:
`scripts/_verify_frame_count.py <wav> <fps>`.

---

## P10 — A fraction-of-a-second of leftover audio plays ~1–2 s AFTER the turn ✅ FIXED (ceil cap, re-applied 2026-06-24)

> **Status: APPLIED.** The `ceil`-the-cap fix below is live in `musetalk_video.py::_advance`. It was
> implemented + verified, reverted-by-preference once (2026-06-23), then **re-applied 2026-06-24** when
> the end-of-turn audio choppiness it leaves was judged worse. Verified 2→0 stranded audio chunks.

**Symptom.** After the voice finished, ~1–2 s of silence, then a brief (<~0.1 s) fragment of the
turn's audio played. Worse after `MUSETALK_END_TAIL_FRAMES` was raised to 10.

**Root cause.** Steady mode releases each buffered audio chunk paired with the rendered frame whose
time covers it (`musetalk_video.py::_advance`/`_emit_pair`). `_advance` capped the release cursor at
`audio_cap = int(audio_clock·fps) + 1` — **`int()` = floor**. A turn's audio length is rarely a whole
number of frames (13.56 s × 12 = 162.72), so the last real frame paired audio only up to
`floor(162.72)/12 = 13.5 s`, stranding the final **sub-frame** (0.06 s). The same floor cap stopped
the END_TAIL frames from releasing it, so it waited for `_drain_audio()` at the `video_end` marker —
which the server delays by END_TAIL (10 frames = 0.83 s) + `idle_grace` (0.3 s) ≈ **1.1 s**. So the
stranded sub-frame played ~1–2 s late as a blip. (P9's frame-count fix made the lips end on time but
left this sub-frame remainder, which this exposed.)

**Fix (applied).** `audio_cap = ceil(audio_clock·fps) + 1` (`musetalk_video.py::_advance`).
`ceil` makes the trailing frame reachable, so the last sub-frame of audio releases paired with the
trailing/END_TAIL frame (~one frame later, contiguous) instead of waiting for the delayed drain. Costs
one frame of look-ahead, which only binds at end-of-turn (mid-turn the binding cap is the rendered-frame
count) and which TTS — running ahead of real-time — has already buffered. Verified with a deterministic
repro (a 13.56 s turn stranded **1** audio chunk to the drain before, **0** after) and on the live path.

## P11 — With echo-guard on, voice stops triggering after the first turn (must type) ✅ FIXED (default flipped, 2026-06-23)

**Symptom.** Speaking a turn produced no response — the user had to type into the client for it to
work. Only after a bot turn; the first interaction could work, then the mic went dead.

**Root cause (a 3-way interaction, all pre-existing).** Echo-guard uses pipecat's
`AlwaysUserMuteStrategy`, which mutes the user on `BotStartedSpeakingFrame` and unmutes on
`BotStoppedSpeakingFrame`. (1) In **steady** sync the voice is held/released *late* (paced to video),
so the output transport sees audio arrive **after** the per-turn `TTSStoppedFrame` → it fires a
**second `BotStartedSpeaking`** right after the early unmute → re-mutes the user. (2) The screech fix
raised `BOT_VAD_STOP_FALLBACK_SECS` to 600 s, so the transport's audio-gap `BotStoppedSpeaking`
never fires afterward. Net: after a turn the mute state machine is left `_bot_speaking=True` with no
unmute → **mic stuck muted**, so STT gets no audio and no turn triggers. Typing bypasses the audio mute.
Confirmed in the log: `... user is now unmuted` (on TTSStopped) immediately followed by
`... user is now muted` while the avatar was still rendering the tail, with no unmute after.

**Fix.** Flip the default to **`ECHO_GUARD=0`** (barge-in; no mute strategy, mic always live) in
`pipeline/config.py`. Lowering `BOT_VAD_STOP_FALLBACK_SECS` was rejected (it reintroduces the P3
screech). Verified: with `ECHO_GUARD=0` a synthesized voice turn triggered cleanly
(`User started speaking` → LLM → `TTFO {count:1, pass:True}`) with **no** mute events. Tradeoff: the
mic is always live → use headphones / OS echo cancellation. `ECHO_GUARD=1` remains valid **only** with
`MUSETALK_SYNC_MODE=live`, where bot-speech framing tracks correctly. Proper half-duplex-in-steady
support (mute on `TTSStarted`/unmute on `TTSStopped` instead of the transport frames) is a future option.

---

## P12 — End-of-turn mouth "snaps" shut (choppy close) ✅ FIXED (2026-06-27, free-run close crossfade)

> **THE FIX (2026-06-27).** A client-side pixel **cross-dissolve** from the last spoken frame -> the
> rest pose, delivered **free-running** (untagged, paced at fps, like the idle loop) for the duration
> of the close — `musetalk_video.py::_play_close_fade`, fired at `video_end`, gated by
> **`MUSETALK_CLOSE_FADE_FRAMES`** (5 = ~0.42s @12fps). Two earlier 2026-06-27 attempts on the way (both
> measured on the WebRTC delivery path, kept here as the record):
> 1. **Rest-pose swap** (`_build_neutral_rendered`, hold a VAE-rendered closed-mouth frame): cut the
>    *domain* pop ~45% offline but the mouth still shape-snapped in one frame. Reverted.
> 2. **Audio-paired crossfade** (each close frame paired with a frame of trailing silence through the
>    steady `_emit_pair` path): correct in principle, but the `_advance` **audio-cap** (which stops
>    video running ahead of voice) **strands** the close frames whenever the render ran behind (video >
>    audio, common on the shared GPU) -> still snapped. So the close must NOT be audio-paired.
>
> The shipped free-run delivery sidesteps both: it's the **"live during the close"** idea — video
> free-runs for just the closing frames (it's still steady through the speech). Requires two supports:
> **(a)** `MUSETALK_END_TAIL_FRAMES=0` so the last buffered frame is the last *spoken* frame, not a
> neutral tail (the crossfade source); **(b)** the server now settles `last` to the **neutral** rest pose
> when idle even with `END_TAIL=0` (`musetalk_server/app.py` pump `elif not sp: last = neutral_frame()`)
> — END_TAIL>0 used to do that implicitly. A `_suppress_until` window drops server idle frames during
> the playout so they can't preempt the dissolve (the burst-flush collapse below). **Verified on the
> delivery path:** the end mouth-to-rest distance now ramps `6.04 -> 2.75 -> 1.85 -> 1.02 -> 0.31 -> 0.07`
> over ~5 frames (snap-index 0.92 -> 0.58) instead of one 6.6 -> 0.8 step.

**Symptom.** When the voice finishes, the avatar's mouth doesn't ease closed — it cuts in one frame
from the last spoken pose to the resting face. Reads as a choppy/abrupt end.

**Root cause (proven, two layers).**
1. **It's a domain jump, not a mouth-shape jump.** Every MuseTalk frame is VAE-rendered; the
   neutral/idle frames are the **original avatar photo**. A rendered frame sits **~5 px** (mouth-region
   mean) from the photo *even when its mouth is also closed*. Measured: rendered close frames stayed
   ~4.9 from neutral, then snapped 4.9→0 in one frame at the cut. So feeding **silence** through
   MuseTalk to "ease it shut" does nothing — silence frames are still rendered, still ~4.9 from the
   photo. Only blending the **pixels** across the boundary (a crossfade) bridges it.
2. **In steady mode the crossfade can't be DELIVERED.** A crossfade (last spoken frame → neutral) is
   smooth at the server, but steady uses pipecat's **non-live** transport, where video frames are only
   spaced out by the **real-time audio frames interleaved between them** (the audio write *is* the video
   clock; see `base_output.py` — `sync_with_audio` images go to the audio queue and a separate task
   draws the current image at `video_out_framerate`). The close has **no audio after the voice**, so the
   trailing frames have nothing to clock them. Three delivery attempts, each verified failing at the
   **WebRTC delivery path** (not just the server):
   - **burst flush** → transport overwrites the current image N times before the draw task samples it →
     only the last (neutral) frame survives → collapses back to the snap (delivered diff 4.47→0.55 in
     one frame).
   - **`asyncio.sleep` pacing** (push one frame per 1/fps) → drifts vs the transport's independent draw
     clock → skipped/duplicated frames, jitter (delivered diff-vs-prev ~3.0, non-monotonic).
   - **silent-audio pairing** (one frame of silence per close frame) → the transport re-buffers/re-chunks
     audio on its own boundaries, dissolving the per-frame pairing → open mouth froze ~0.9 s then snapped.

**Why it works in `live` mode.** `live` sets `video_out_is_live=True`: video runs on its **own** clock
(`_video_queue`/`_video_is_live_handler`), independent of audio. The server's crossfade frames then
stream through at the server's rate → smooth close. So a smooth close is achievable **only** in live —
but live trails the lips ~0.75 s behind the voice during speech, which the user rejected.

**The 2024-era conclusion ("not deliverable in steady") was too strong — here's the seam.** The three
attempts above all tried to deliver the close while still *coupled* to the audio (burst into the
audio-clocked image slot, or pair with silence). The seam is to deliver the close **decoupled** from
audio: push the dissolve frames as plain untagged `OutputImageRawFrame`s paced at fps — the SAME path
the idle/breathing loop already uses successfully in steady. The old "`asyncio.sleep` pacing jittered"
attempt was close but pushed a *server* crossfade through a racing path; doing it client-side at
`video_end`, landing exactly on the rest pose, with the `_suppress_until` guard against server-idle
preemption, delivers cleanly. Net: **steady through the speech (synced lips), video free-runs only for
the ~0.4s close.**

**Why `MUSETALK_CLOSE_FADE_FRAMES=0` is still valid.** Set it to 0 (and restore `END_TAIL` > 0) for the
old clean snap. The P10 ceil audio-blip fix is independent and kept.

**Why a pure `live`-mode close still works too** (unchanged): `live` sets `video_out_is_live=True` so
video always runs on its own clock — a smooth close for free, at the cost of the ~0.75s lip trail during
speech the user rejected. The free-run close above gives the same close benefit without that trail.

**Tooling:** `scratchpad probe_close.py` drives a real turn and SAVES the received WebRTC frames
(`VideoFrame.to_ndarray`) so the close can be judged on the **delivery path** (the offline server
capture bypasses the transport and cannot see a delivery collapse). Judge the close where the browser
sees it.

---

## P13 — MOSS-TTS "delay between sentences" ⚠️ NOT RESOLVED (2026-06-29; streaming+eager helped TTFB but the felt latency is still bad — and got WORSE)

> **HONEST STATUS (2026-06-29, end of session).** The streaming + eager changes below cut the *isolated*
> per-request TTFB (benchmarked ~0.4 s), but **the user reports the between-sentence delay is still there
> and overall latency is now WORSE.** So the isolated-TTFB win did NOT translate to a smooth live
> conversation — do not trust the "fixed" framing. **Leading hypothesis (untested): CPU contention.**
> This session also moved the **LLM onto a CPU-pinned local Ollama** and was running the memory harness,
> the memory-sim, and the weather mock — all on CPU — while the GPU ran CosyVoice-vLLM + MuseTalk. The
> original smooth baseline used a **cloud** LLM, leaving the CPU free. The most likely culprit is the
> machine being CPU-saturated, not the TTS engine. **Plan: revert `.env` to the baseline (cloud LLM +
> CosyVoice) next session and re-measure end-to-end TTFO before touching MOSS again.** The vLLM-Omni path
> (no torch.compile, GPU-served) remains the real fix for MOSS if it's pursued.

**Symptom.** With `TTS_PROVIDER=moss`, the avatar took a long beat before each sentence — felt like a
big lag, "is it the bigger model?". (Still present after the changes below.)

**Root cause (measured, not guessed).** Two separate things, neither the parameter count:
1. **The first server was non-streaming.** It called MOSS's `inferencer.generate()` (whole sentence),
   THEN streamed the finished PCM. Measured: time-to-first-audio **8.55 s** ≈ total wall **8.58 s** —
   they were identical, i.e. the avatar waited for the entire sentence before any sound. (The 1.7B size
   only makes steady-state RTF ~1.6, a minor factor.)
2. **Once streaming, `torch.compile` recompiled per sentence-length.** The streaming rewrite
   (`MossTTSRealtimeStreamingSession` + `AudioStreamDecoder`: push_text → decode → drain → flush) dropped
   warm TTFB to ~0.4 s — but the **first** time it saw each new token-length it recompiled **3–40 s**, and
   a real reply has many lengths, so the spikes landed *between sentences*.

**The fix.** (a) Stream the first chunk (the rewrite above). (b) Run **eager** — the server defaults
`TORCHDYNAMO_DISABLE=1` (override `MOSS_COMPILE=1`). Eager has **no recompiles**: every sentence is a
consistent **~0.35–0.5 s** TTFB (vs compiled's 0.25 s warm but 3–40 s spikes). Verified across varied
lengths: worst-case TTFB 0.53 s, zero spikes. Tradeoff: eager's long-sentence steady-state is a bit
slower; the both-fast-and-no-spikes path is **vLLM-Omni** (MOSS supports it natively — the next step).

**Install gotchas hit along the way** (all in the server docstring): MOSS's streaming codec path needs a
**C compiler** for triton (`CC`/`CXX` → `conda install -c conda-forge gcc gxx`); `torchcodec` couldn't
`dlopen` until **ffmpeg pinned to 7.1** (8.x is too new) + **`nvidia-npp-cu12`** installed +
**`LD_LIBRARY_PATH`** covering torch/lib + the env lib + the `nvidia/*` pip libs. Files:
`local_services/moss_server/app.py`.

---

## P14 — Config-panel "Restart pipeline" showed "restart request error" ✅ FIXED (2026-06-29, native Win32 kill)

**Symptom.** Clicking **Restart pipeline** in the config panel returned *"restart request error"* in the
browser instead of restarting.

**Root cause.** The restart handler shelled out to a **PowerShell cmdlet** (`Get-NetTCPConnection | Stop-Process`)
to kill `:7860`. On this box under CPU load (the memory-sim auto-run was grinding a CPU model at the
time), **PowerShell — and `taskkill`, and `tasklist` — take tens of seconds to even start** (watched
`taskkill` and a `Get-NetTCPConnection` each hang 20–30 s+ while plain `netstat` and Python returned
instantly). The `subprocess.run(..., timeout=20)` blew its timeout, raised `TimeoutExpired` **inside the
request handler**, the handler died without sending a response → the browser saw a dropped connection =
"restart request error".

**The fix.** Kill the pipeline PID with a **native Win32 `OpenProcess`/`TerminateProcess`** (via
`ctypes`) — it returns instantly even when `taskkill`/PowerShell are hung — after finding the PID with
fast `netstat`. And wrap the whole restart in try/except so it **always returns JSON** (`{"ok":…, "message":…}`),
never a dropped connection. Verified: restart now completes in **~13 s** end-to-end (kill → relaunch →
wait for `:7860` to bind) and reports `"pipeline restarted (bound :7860)"`. Lesson: **on this machine,
prefer `netstat` + native kill over `taskkill`/PowerShell for anything time-sensitive** — the Windows
process-management tools are pathologically slow to spawn under load. File: `local_services/config_panel/server.py`.

---

## P15 — Chinese voice starts ~1s later than English ⚠️ NOT RESOLVED (2026-06-30; root cause found, fix conflicts with the avatar)

**Symptom.** Lip-sync looks fine in **English** but in **Chinese** the voice starts noticeably late and
feels out of step ("avatar runs first"). Judged on a remote Tailscale browser (a trustworthy A/V judge,
not the RDP window).

**Root cause (measured at the component boundary, not guessed) — it is the TTS, not the LLM or the avatar.**
CosyVoice's time-to-first-audio is **~2.0–2.75 s for Chinese vs ~1.0–1.5 s for English** (pipeline
`CosyVoiceTTSService TTFB` logs; consistent every turn, not a cold-start). The library's own per-segment
log (`yield speech len X, rtf R`) shows why: the **first streaming chunk is ~4.36 s of audio for Chinese
vs ~2 s for English at the SAME rtf (~0.5)** — a bigger opening chunk takes longer to generate before it
is yielded. In `steady` sync this TTFB stacks on top of the ~2 s avatar render-readiness hold, so the zh
voice starts ~4.4 s after you stop vs ~3.2 s for en. **Ruled out by experiment:** NOT text normalization
(the `wetext`/`ttsfrd` frontend isn't installed in the WSL `cosyvllm` env → `text_frontend=''`), and NOT
the zero_shot-vs-cross_lingual inference path (forcing zh through `inference_cross_lingual` did **not**
change TTFB). The LLM is also ruled out (its TTFB is similar/lower for English).

**The fix that works in isolation BUT regresses the system — do NOT ship.** `COSYVOICE_FIRST_HOP=5` (the
existing knob in `CosyVoice/cosyvoice/cli/model.py`, default `0`=25 tokens) emits the first audio after
fewer speech tokens → zh TTFB **2.3 s → 1.25 s** (verified). But it makes CosyVoice run **many more small
flow+vocoder GPU inferences at turn start**, which on the single shared 16 GB GPU **contend with MuseTalk's
render and starve the avatar — lips-start jumped ~2 s → ~8 s** (much worse overall). Reverted.

**The real constraint.** On one GPU, **fast Chinese TTS streaming and a smooth avatar are in direct
conflict**: anything that makes the audio come out sooner adds GPU work that starves the avatar render.
This is the same reason `COSYVOICE_PACE_RATE` deliberately throttles voice production. A genuine fix needs
a **dedicated avatar GPU** or a lower-contention TTS path — not a settings tweak. Left at the default
(zh ~2.3 s TTFB) so the avatar stays smooth.

**Shared-GPU restart gotcha (learned the hard way this session).** CosyVoice's vLLM must load **before**
MuseTalk. Restarting cosyvoice while MuseTalk already holds ~5 GB crashes vLLM with
`ValueError: No available memory for the cache blocks` at `gpu_memory_utilization=0.3` (its runtime
overhead ~5 GB needs the room; raising util then trips `Free memory … less than desired`). Recovery =
restart the **whole stack in order**: stop all → start cosyvoice on the near-empty card (`run_vllm_server.sh`)
→ then MuseTalk + pipeline via `scripts/run.ps1`. (Alternatively cap vLLM with `max_model_len` so it fits
second — TTS sequences are short — but the ordered restart is the baseline.) Healthy VRAM with all three
loaded ≈ **300–400 MB free**. Full detail: `project-visualllm-zh-tts-latency-gpu-contention` memory.

## P16 — Avatar lips drift progressively behind the voice on long turns ✅ FIXED (2026-07-01, TensorRT render is now the baseline)

**Symptom.** On a **long** (multi-sentence) reply the lips fall further and further behind the voice; a
short reply looks in step. "Frames don't equal the audio" — the delivered video runs seconds longer than
the audio.

**Root cause (measured live, driving the MuseTalk server with real WAVs at prod fps, offline — no
CosyVoice/WebRTC).** Two things were conflated:
- The **rendered lip-frame count** (server `video_clock`) **already equals `audio_sec × fps`** (±1, the P9
  ceil pad). The lips are NOT missing content and don't finish early.
- The **extra** frames are the pump's **HELD/duplicate frames** — it repeats the last frame to keep the
  WebRTC track continuous whenever render dips below fps. They pad the timeline, carrying no new mouth
  movement.

Per 8-frame segment on the PyTorch path: gpu 259ms + composite ~120ms ≈ **389ms** vs the 667ms real-time
budget @12fps (~1.7× headroom). So **alone** MuseTalk barely drifts (a fixed +0.36s startup offset at any
length). The drift only becomes **length-scaling** once the effective render rate drops below fps — which
happens under **CosyVoice's shared-GPU contention**. Proven with a GPU compute hog (100% util, CosyVoice
stand-in): PyTorch drifted `+0.37s (2.9s reply) → +1.35s (5.5s) → +3.94s (13.6s)`, render ~9fps. Formula:
`drift ≈ audio_len × (1 − render_fps/fps)`. So the long-turn drift is contention, **not** the frame math.

**Fix — TensorRT render path, now the default (`MUSETALK_TRT=1`, merged to `main`).** TRT engines replace
the UNet + VAE-decoder GPU calls (`render_segment`): gpu 259→**168ms**, composite ~120→**78ms**,
total/seg 389→**~255ms** (~1.5×), lifting headroom 1.7×→**2.6×**. **Under the SAME 100% contention that
drifted the PyTorch path +3.94s on the 13.6s reply, TRT holds the drift flat at +0.36s at every length**
(held frames 50→4). Engines are GPU/driver-specific (~1.75GB, gitignored) — build once with
`local_services/musetalk_server/trt_build.py`; any load failure silently falls back to the proven PyTorch
path. Next cheap lever (no 2nd GPU): the composite is CPU PIL blending (~31% of render even with TRT) —
move it to the GPU. Structural fix remains a **dedicated avatar GPU**. Full detail:
`project-visualllm-musetalk-trt-drift-fix` memory.

## P17 — Per-frame composite was CPU-bound (~31% of render even with TRT) ✅ FIXED (2026-07-01, GPU composite, opt-in)

**Context (the P16 "next cheap lever").** After TRT, each rendered frame still ran `get_image_blending`
(PIL crop/paste + cv2) on the **CPU** to alpha-blend the rendered mouth back onto the base portrait and
downscale to the output size. Measured at ~**68 ms per 8-frame segment** with TRT on (`MUSETALK_PROFILE=1`)
— about a quarter of the ~255 ms/seg render, and it also forced a GPU→CPU copy of the VAE output every
frame. On the shared GPU that CPU time is headroom lost to nothing.

**Fix — `MUSETALK_GPU_COMPOSITE=1`: do the blend + downscale on the GPU in torch.** With the TRT path the
VAE output is **already a GPU tensor**, so the composite runs there with no extra transfer:
`_composite_gpu` (`app.py`) bilinear-resizes the mouth, alpha-composites it into the crop_box region with
the precomputed mask, downscales to `MUSETALK_SIZE` with `mode="area"`, and transfers only the final
frame. Base frames + masks are uploaded once at load (`_init_gpu_composite`).

**Result (measured, `_drive_frames.py` + `MUSETALK_PROFILE=1`, prod fps=12/SIZE=256, clean no-contention
drive on the 13.56s reply):** per 8-frame segment, gpu (UNet+VAE) **~170 ms unchanged** + composite
**~73 ms → ~11 ms** (~6.6×) → total **246 → 182 ms**, i.e. **~26% off render, render ceiling ~33 → ~44
fps**. **Output is pixel-identical:** SSIM **1.0**, max **≤1 LSB** vs the CPU path across smooth,
random-high-frequency and checkerboard mouth content (the blend is content-independent, so the 1-LSB gap
is just rounding). No render errors over a full turn.

**Honest read of the benchmark — it does NOT reduce A/V drift *today*.** At 12 fps both configs sit far
above the 667 ms/seg real-time budget (246 and 182 ms), so the paced A/V metrics are **identical**: drift
flat +0.69 s at every length (2.88/5.48/13.56 s), 7 held frames — **even under a verified 100% GPU
contention hog** (TRT alone already holds ≥12 fps, so the composite saving isn't the deciding factor).
The value is **reserve**: (1) the render ceiling rises 33→44 fps — headroom for heavier/real-bursty
contention, higher fps/resolution, or a weaker/dedicated GPU; (2) it **frees the CPU** (the CPU PIL blend
is gone), which the live pipeline needs for STT / pipecat / LLM-streaming. That CPU relief is exactly what
the offline render-isolation test can't see and what the **live call (#1)** is the judge of. So: strictly
not worse (pixel-identical, drift unchanged), with real headroom banked. VRAM change is negligible
(base-frame + mask tensors, tens of MB).

**Gates / safety (mirrors how TRT landed).** Code default **off** (opt-in, public-repo-safe); this box's
`.env` sets `1` and `run.ps1` propagates it. **Only active with `MUSETALK_TRT=1`** — the PyTorch path's
`recon` is CPU numpy, so there it logs "ignored" and keeps the CPU composite. If any `crop_box` runs
off-frame (an off-center/edge portrait) it **disables itself and falls back to the CPU path** (logged).
One-time cost: the first turn after a server start pays ~100 ms of `F.interpolate` CUDA-kernel init
(seen as elevated composite on the first 2 segments), then it settles. **Still open:** the composite
`empty`/`clone` allocations could be pre-allocated, and the base-frame `clone()` per frame could be an
in-place write into a scratch buffer — micro-optimizations, not needed yet. Structural fix is still a
**dedicated avatar GPU**.

## P18 — Chinese voice "halting/broken" + avatar keeps moving after the voice ✅ FIXED (2026-07-02, RAS restored in vLLM + fluid "pro" voice)

**Symptom.** With `LANGUAGE=zh`, the Chinese voice sounded broken ("like autism speaking" — long unnatural
pauses) and the avatar kept animating after the words ended. **English was perfect.** Both are Chinese-only
and are ONE bug in the TTS, not the avatar/steady/GPU. (Distinct from P15, which was zh *first-chunk latency*.)

**Root cause (proven offline, no pipeline/WebRTC — hit `/tts/stream` directly and analyze the CONCATENATED
PCM, never per-chunk).** CosyVoice2's autoregressive speech-token LLM runs on **vLLM** here (the P-era
latency fix), and the vLLM decode path does its OWN sampling — it **lost CosyVoice's Repetition-Aware
Sampling (RAS)**. Native RAS (`cosyvoice/utils/common.py::ras_sampling`): nucleus (top_p=0.8/top_k=25), then
if the sampled token appears in the last `win_size`=10 decoded tokens, ban it and resample — the guard that
stops the model looping on the silence token. vLLM's `inference_wrapper` used plain `SamplingParams(top_k=25)`
= no guard. So the LLM intermittently loops on silence: the SAME short zh sentence came out ~4s clean one run
and **~12s with ~5s of dead silence** the next (~40% of zh runs); English rock-stable. That silence is BOTH
symptoms — the pauses = "halting" voice; MuseTalk faithfully renders frames for all ~12s (idle mouth through
the silence) while the words are only ~4s = "avatar keeps moving after the voice". zh uses
`inference_zero_shot` (denser/longer token seqs) and hits the loop; en uses `cross_lingual` and doesn't.

**Fix 1 — restore RAS inside vLLM's sampler (the real root-cause fix).** New
`CosyVoice/cosyvoice/vllm/ras_logits_processor.py`: a vLLM V1 `AdapterLogitsProcessor` whose per-request
`(output_ids, logits)` callback sets `-inf` on every token seen in the last `COSYVOICE_RAS_WIN` (=10) OUTPUT
tokens — identical anti-loop effect to native RAS, using OUTPUT tokens only (embeds-safe). Registered via
`EngineArgs(logits_processors=[RasLogitsProcessor])` in `cli/model.py::load_vllm`; `top_p=0.8` added to
`llm.py::inference_wrapper` to match RAS's nucleus. **DEAD END (do not retry):** vLLM's own
`repetition_penalty`/`frequency_penalty`/`presence_penalty` all build a prompt-token bincount, but CosyVoice
feeds `prompt_embeds` (no prompt token ids) → `ScatterGatherKernel index out of bounds` CUDA **device-side
assert** that kills the engine (corrupts the CUDA context → full restart). Verified: was ~40% degenerate →
**48 varied zh runs, 0 degenerate**; tighter + lower-latency than an interim output-guard/retry workaround
(abort_request on a dominant-token window + zh retry-on-empty), which was **reverted** in favor of this.

**Fix 2 — a naturally fluid reference voice (why zh still felt choppy-vs-en after Fix 1).** With the loop
gone, zh still measured gappier than en: **~57% voiced / ~3.8 pauses per sentence vs en ~65% / ~2.5**. Ruled
out: the `speed` knob (1.15/1.3 didn't reduce dur or gaps) and the `cross_lingual` path (57%→59%, marginal).
The real lever is the **reference clip** — `zero_shot` clones its RHYTHM. Swapped the gappy "weather" clip
(`asset/zero_shot_prompt.wav`) for the **MOSS "pro" AI-assistant voice** — found on disk at
`visualllm/assets/moss_pro_ref.wav`, language-confirmed + transcribed via Deepgram (conf 1.00:
"你好，我是你的AI虚拟助手…"), copied to `CosyVoice/asset/pro_ref.wav`, now the default `PROMPT_WAV`/`PROMPT_TEXT`
in `tts_engine.py`. Result: zh → **~64% voiced / ~1 pause per sentence** (fewer than English), smooth with no
trimming. **Correction to an earlier in-session claim:** the choppiness was NOT "inherent to the model" (that
was premature — concluded before testing another voice); it was the reference clip. CosyVoice2 is Chinese-first
and is fine at zh.

**Interim band-aid, now off by default.** Before Fix 2, a streaming pause-compressor
(`TTSEngine._squeeze_silence`, `COSYVOICE_SILENCE_CAP_S`) capped over-long zh silences to match en pacing
(brought zh to 64% voiced). The pro voice makes it unnecessary, so it is **OFF by default**
(`COSYVOICE_SILENCE_CAP_S=0`) but kept as an optional knob for a gappy voice.

**Files (all in `E:\Claude\cosyvoice-local-tts`; the nested `CosyVoice/` is its own git repo):**
`CosyVoice/cosyvoice/vllm/ras_logits_processor.py` (new), `CosyVoice/cosyvoice/cli/model.py` (register the LP),
`CosyVoice/cosyvoice/llm/llm.py` (`top_p=0.8`), `CosyVoice/asset/pro_ref.wav` (new voice clip),
`tts_engine.py` (pro voice defaults + the off-by-default squeeze). **Not yet git-committed.** Baseline knobs:
`COSYVOICE_RAS_WIN` (10), `COSYVOICE_PROMPT_WAV`/`COSYVOICE_PROMPT_TEXT` (pro voice), `COSYVOICE_SILENCE_CAP_S`
(0 = off). **Still open:** human A/V confirmation on a live call; git-commit both repos.

## P19 — TTFO tuning sweep: `COSYVOICE_FIRST_HOP` × `MUSETALK_LEAD_FRAMES` ⚠️ NO WIN — baseline `hop=0, lead=14` stands (2026-07-03)

**Goal.** Find a `first_hop` (and later, jointly, a `LEAD_FRAMES`) value that lowers TTFO **without** degrading
the avatar. Ended with a **negative result**: every candidate that lowered TTFO either got erased live or
introduced choppiness the objective metrics initially missed but the human eye caught. The baseline
(`hop=0`, `lead=14`) is already correct. This entry records the method + all data so it is not re-litigated.

**Two knobs, precisely (read the code, not the memory):**
- `COSYVOICE_FIRST_HOP` (`CosyVoice/cosyvoice/cli/model.py:391`, per-language via `tts_engine.py::_apply_first_hop`):
  the FIRST streaming chunk normally waits `token_hop_len=25` speech tokens (~1s audio) before emitting. `hop<25`
  emits after fewer tokens = a **smaller opening chunk** → lower isolated TTFB but less opening audio. `0`=off=25.
- `MUSETALK_LEAD_FRAMES` (`musetalk_server/app.py:685,729`): the pump **holds the last frame and won't release the
  synced voice until `lead_frames` rendered frames are queued** (`out_q.qsize() >= lead_frames`). It IS the
  synced-start delay, and it is a **mid-turn shock absorber** — the cushion drains on a render dip instead of the
  queue hitting empty (which in steady = a voice pause/stutter).

**Method (how the sweep ran cheaply + correctly).** Live-settable debug endpoints avoided restart-per-value:
`GET /debug/hop?en=&zh=` on the cosy server (hop is read per-request) and `GET /debug/lead?n=` on the MuseTalk
server (lead is read per websocket connection → a fresh pipeline connect picks it up). **Both are TEMPORARY and
were reverted** (source `git`-clean; they linger only in a running process until its next restart). TTFO was
decomposed into **LLM-independent** parts to beat the OpenRouter cloud-hop variance (0.8–7.3s): `raw_TTS` =
`tts→avatar − llm→tts` (the hop effect) and `delay` = `TTFO − tts→avatar` (the lead effect). Choppiness was
measured **server-side** (see the metric note at the end) — the reliable signal.

**Data 1 — isolated TTFB, CosyVoice alone, no avatar (n=6, tight).** GOTCHA: read the stream with a **small**
buffer (`read(1024)`); a 64KB `read()` blocks accumulating multiple chunks and *inflates/masks* TTFB (a first
noisy pass wrongly showed hop=0 as fastest). True TTFB **saturates by hop≈3** for both languages:

| hop | en TTFB | zh TTFB |   | hop | en TTFB | zh TTFB |
|-----|---------|---------|---|-----|---------|---------|
| 0(=25) | 2.62s | 3.04s | | 5 | 1.89s | 2.02s |
| 1 | 1.82s | 1.83s | | 10 | 1.90s | 2.00s |
| 3 | 1.86s | 1.84s | | | | |

**Data 2 — the isolated win INVERTS live (full stack, steady, at the default `lead=14`).** A smaller opening
chunk fills the `lead=14` cushion slower → the synced voice-start is *delayed*, and hop's extra small vocoder
bursts contend with MuseTalk on the shared GPU (erasing even the raw-TTS gain live):

| hop @ lead14 | en live TTFO | zh live TTFO | note |
|-----|------|------|------|
| 0 | ~3.6s ✅ | ~3.68s ✅ | biggest opening chunk |
| 3 | 4.42s | 4.22s | starved start |
| 5 | ~4.0s | ~5.78s | worse; zh render freeze 162–202ms |

**Data 3 — high hop (<25) at `lead=14` is a plateau+cliff, not a slope.** hop 15–22 return to ~hop=0 behavior
(the chunk is still big enough to fill the cushion); only hop≤10 starve. hop=20 TTFO ~3.42s ≈ hop=0 ~3.57s — a
tie within noise, never a real win, and nearer the starvation cliff.

**Data 4 — the 24-cell grid `hop{0,2,5,10} × lead{14,10,8,6,4,2}` (LLM-independent).** The starvation is a
**`lead=14` artifact**: at low hop the small chunk can't fill a *big* cushion. Synced-start `delay` has a **cliff
between lead 10 and 8** — at lead≤8 the delay collapses to ~0.4–0.6s for EVERY hop:

```
delay(hop,lead):     lead=14  10    8     6     4     2
  hop=0 (big chunk)    0.79  0.74  0.52  0.55  0.34  0.50   <- big chunk: low delay at any lead
  hop=2                2.05  1.85  0.46  0.37  0.42  0.56   <- CLIFF between lead 10 and 8
  hop=5                1.82  1.82  0.61  0.48  0.44  0.59
  hop=10               1.58  2.70  0.61  0.63  0.38  0.56
```

**Data 5 — verified candidates, n=4 (COMBINED = raw_TTS + delay, LLM-independent).** The low-hop+low-lead corner
beats baseline by ~0.6–0.9s on *latency*:

| cell | COMBINED | freeze | vs baseline |
|------|----------|--------|-------------|
| baseline hop0/lead14 | 2.89 | 110–154ms | — |
| hop5/lead6 | 2.05 | 106–113ms | −0.84s |
| hop2/lead4 | 1.98 | 106–**171ms** | −0.91s |
| hop0/lead6 | 2.27 | 108–112ms | −0.62s |

**Data 6 — offline contention validation (13.6s reply, `_drive_frames` + `_gpu_contention_hog.py` N=4096, 100%
GPU).** `lead=6` == `lead=14`: 162 rendered, 4 held, +0.357s drift, 11.8fps — **no underflow.** This PASSED but
was **MISLEADING**: a *steady* matmul hog does not replicate CosyVoice's *bursty* vocoder contention, and this
test had no CosyVoice at all.

**Data 7 — the DECIDING test: live choppiness, server-side `[chop]` held% (n=4).** Only the **combination**
`hop=5 + lead=6` is choppy; either knob alone is baseline-smooth:

| config | held% (smooth ≈17%) | verdict |
|--------|--------------------|---------|
| baseline hop0/lead14 | 17.6 [14–20] | smooth (reference) |
| **hop5/lead6** | **36.5 [35–37]** | **CHOPPY** (matches the human eye) |
| hop0/lead6 | 17.2 [17–21] | smooth |
| hop5/lead14 | 16.9 [14–18] | smooth |
| hop0/lead10 | 17.3 [15–20] | smooth |

**The mechanism (why the combo is choppy, and why my offline validation missed it).** `hop=5` makes CosyVoice
emit **more, smaller** vocoder bursts (extra shared-GPU contention spikes during the turn); `lead=6` is a **thin
cushion**. Either alone survives — a big cushion absorbs the bursts, OR big chunks (hop=0) don't create the
bursts so a thin cushion suffices. **Together**: bursts + thin cushion → mid-turn render underflow → held frames
→ choppy. Data 6 missed it because the steady hog isn't bursty and ran without CosyVoice.

**Live A/V verdict (human).** `hop=5,lead=6` → user: **"choppy."** Confirmed by Data 7. Retesting `lead=14`
surfaced a *separate* symptom — **"avatar starts before the voice."** The log shows why and that it is NORMAL
steady behavior (present at clean baseline): audio arrives ~2s **ahead** of video (`[musetalk sync] hold=2.07s,
audio 4.2s, video 2.2s, vbuf=0`) and steady is **video-master**, so it **holds the voice** waiting for the
render — when the render lags, the avatar's frames play while the voice is withheld → lips-first. **The lever
for lips-first is `MUSETALK_SYNC_MODE` (steady↔live), NOT hop/lead.** `live` = voice instant, lips trail ~0.75s.

**Conclusion — ship nothing; baseline `hop=0, lead=14` stands.** Every TTFO "win" came from starting the voice
sooner, which in steady = shrinking the hold/cushion that keeps lips and voice together. The isolated hop TTFB
gain is real but erased live (synced-start fill + shared-GPU contention). **This UPDATES P15**: the "low hop
starves the avatar" finding was specifically a `lead=14` artifact (small chunk vs big cushion) — but lowering
the cushion to recover it re-introduces choppiness via bursty contention, so P15's practical verdict (don't ship
hop) is unchanged, with a cleaner mechanism. A genuine TTFO win still needs a **dedicated avatar GPU**, not a
settings tweak.

**Measurement lessons (load-bearing for the next tuning attempt).**
- **Choppiness must be measured server-side.** The pump *knows* when it holds a frame (`out_q` empty while
  `playing` = a real render underflow) vs a real/idle frame. A temporary counter there (`held_playing`, logged
  as `[chop] held%` at `video_end`) cleanly separated smooth (17%) from choppy (36%).
- **What did NOT work:** (a) WebRTC received-frame duplicate detection — VP8 lossy decode blurs "exact duplicate"
  and natural inter-sentence mouth-stillness looks identical to an underflow freeze; (b) `freeze_ms` (max single
  gap) — too coarse, misses sustained low-grade micro-stutter; (c) offline steady-hog validation — not bursty,
  no CosyVoice.
- **Always A/B on the LIVE full stack (real CosyVoice contention), and let the human eye be the final gate** —
  it caught choppiness the first (offline) validation certified as safe.

**Files.** All temporary instrumentation reverted (both `/debug` endpoints + the `[chop]` counter);
`local_services/musetalk_server/app.py` `git`-clean. No config change shipped. `.env` unchanged
(`COSYVOICE_FIRST_HOP` unset/0, `MUSETALK_LEAD_FRAMES=14`). Full record: this entry +
`project-visualllm-ttfo-first-clause-split` memory.

## P20 — FP8 quantization of the render UNet ⚠️ DEAD END (2026-07-03, proven; sm_120 has no FP8-conv kernels). Handoff: Lever 1 (GPU priority) + Lever 3 (stagger)

**Goal.** Fix the "some turns the avatar's mouth moves before the voice" symptom at its root. Root cause (same
shared-GPU wall as P15/P16/P19): the ONE 16GB card runs CosyVoice (TTS gen, in WSL) + MuseTalk (render,
Windows); at **turn start** CosyVoice's opening vocoder burst and MuseTalk's **first** render segment collide,
the render loses GPU time → in `steady` (video-master) the voice is held while the already-rendered lip frames
show → "avatar before voice". TRT's mid-turn headroom does NOT cover the *first-frame* render (P15). "Lever 2a"
= shrink the UNet's GPU cost with FP8 so the starved first segment finishes sooner.

**Result = REJECTED. Quality perfect, 4.5x SLOWER.** modelopt FP8 PTQ of the UNet: **SSIM 0.99997 vs fp16**
(max pixel diff ≤17/255 — the quantization MATH is correct), but **fp16 44.4ms → fp8 200.5ms per UNet batch**.
The engine built but ran ~4.5x slower because TensorRT could not compile FP8 kernels and silently fell back to
running the Q/DQ as pure overhead on fp16.

**Root cause of the slowness (proven, not guessed): TensorRT 10.13.3.9 has no FP8-convolution kernels for
Blackwell sm_120.** Build log: **82 skipped FP8 tactics + 30 `NVRTC Compilation failure` /
"No matching rules found for input operand types"** on the Conv/MatMul weight `QuantizeLinear` nodes (Myelin
codegen `nvrtc_compile.cpp` CHECK failure = the compiler literally cannot generate FP8 conv code for this
arch). **Ruled out the competing "scale format" hypothesis with a decisive tiebreaker:** modelopt's ONNX FP8
path hardcodes `per_channel=True` (fp8.py:302; it does INT8 PTQ then int8→fp8, inheriting per-channel weight
scales → `f16[320]` vectors), and NVIDIA's forum says TRT FP8-conv wants **per-tensor**. Monkey-patched
`quantize_static(per_channel=False)` (`trt_quant_fp8.py --per-tensor`) → the scale became a scalar `f16[]`
(format now exactly what TRT wants) but the failure was **byte-identical (same 82 skipped + 30 NVRTC + 4.5x
slower)**. So per-channel was a RED HERRING; it's purely the sm_120 kernel gap. Corroborated: NVIDIA/TensorRT
issue #4715 (documented sm_120 Myelin kernel-library gaps; even TRT 10.13.3 "fails to init on sm_120 with a
different error"), and the "FP8 conv tactics not used" forum thread (finicky even on Hopper sm_90 where support
is mature). **No scale-format or code change on our side can fix it — retry only after a newer TensorRT ships
sm_120 FP8-conv kernels, then rerun `trt_quant_fp8.py --per-tensor`.**

**Toolchain notes (reuse when FP8 is retried):** use modelopt's **ONNX PTQ** (`modelopt.onnx.quantization.quantize`,
`quantize_mode="fp8"`), NOT torch-quant + `torch.onnx.export` (torch 2.11's TorchScript FP8 symbolic chokes:
`amax` comes through as a graph Param; `export_torch_mode` + opset tweaks don't fix it). ONNX PTQ works on the
existing fp16 `unet.onnx`. GOTCHAS: (1) pass `calibration_method="max"` — the default `entropy`/histogram is
the slow INT8 method (ran >1hr on 876 tensors before I killed it; `max` = one pass, ~5-8min total). (2) the
`CalibrationDataReader` needs `get_next` AND `get_first`. (3) `calibration_eps=["cuda:0"]` (onnxruntime-gpu).
Script: `local_services/musetalk_server/trt_quant_fp8.py` (calibrates on the real reply `output/_mic_drive.wav`,
builds `unet_fp8.engine`, validates SSIM + per-batch GPU time vs the fp16 engine).

**Env damage + repair (the modelopt install is riskier than pip's dry-run shows):** `pip install
nvidia-modelopt[onnx]` into the `musetalk` env silently upgraded **numpy 1.26.4 → 2.2.6** (pulled by
`cupy-cuda12x`), which **broke cv2 + face_alignment** (`_ARRAY_API not found`) — the server would crash on
restart. Repaired mid-session (uninstall cupy + pin numpy). After FP8 was ruled out, the env was **fully
rolled back** to `logs/_env_snapshot_pre_fp8.txt` (24 added pkgs uninstalled, onnx restored 1.21→1.22) —
verified **bit-identical** to the snapshot, all imports clean. **Take a pip-freeze snapshot before any future
modelopt install.** Original fp16 `unet.engine` (06-30) never touched; dead `unet_fp8.*` deleted.

**Live measurement (`scripts.measure`, system python, real driven turn) — the shared-GPU wall, quantified:**
an 11s reply rendered at a sustained **12.1fps** but the video ran **~2.7s behind the voice all turn**
(`[musetalk sync] hold=2.71s`, `[avatar timing] lips start +2.88s after voice`). This **contradicts P16's
offline "TRT holds drift flat at +0.36s"** — that was measured WITHOUT CosyVoice; under real (bursty) CosyVoice
contention the turn-start hold is ~2.7s. Confirms (again, per P19) that this cost only appears on the LIVE full
stack.

**HANDOFF — two untried non-hardware levers (both directly attack the turn-start GPU collision):**

- **Lever 1 — GPU stream PRIORITY (let the render cut the line).** Mark MuseTalk's render as a high-priority
  CUDA stream so its kernels preempt CosyVoice's during the collision; the avatar holds ≥12fps, CosyVoice
  finishes a few ms later (RTF<1, huge slack). **Impl:** `musetalk_server/app.py` / `trt_runtime.py` — run
  `render_segment` under `hp = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[0])` (most
  negative = highest), `with torch.cuda.stream(hp):` around the render so `trt_runtime`'s
  `torch.cuda.current_stream()` (feeds `execute_async_v3`) uses it, then `hp.synchronize()`. **Honest caveat =
  the whole risk:** CosyVoice (WSL) and MuseTalk (Windows) are separate GPU processes under the Windows **WDDM**
  scheduler, which does NOT strongly honor cross-process stream priority (CUDA MPS would, but is Linux-only, not
  on Windows). **Cheap to test, uncertain to bite across the WSL/Windows boundary** — measure live `hold=` /
  `lips start +Xs` before vs after; if unchanged, WDDM ignored it and this lever is dead.

- **Lever 3 — STAGGER the two bursts.** They collide only because both peak at the same instant; CosyVoice's
  burst is short + front-loaded. **(a) [preferred, lower risk]** throttle CosyVoice's generation nearer
  real-time at turn start so it stops front-loading the GPU — lever on `COSYVOICE_PACE_RATE` (currently 1.3, in
  the cosyvoice repo server), maybe a dedicated turn-start throttle, to leave the avatar's first render room; no
  sync surgery. **(b)** on `musetalk_video.py`, delay feeding the FIRST render segment ~150–250ms so CosyVoice's
  opening burst clears first (trade: adds that fixed delay to lip-start). Watch TTS TTFO doesn't cross the 8s
  target.

  **Measurement protocol for BOTH (don't repeat P19's offline-hog mistake):** A/B on the LIVE full stack with
  real CosyVoice — `python -m scripts.measure --mic output/q_long.wav` — read `[musetalk sync] hold=` +
  `[avatar timing] lips start +Xs` from `logs/pipeline.log`; human eye is the final gate; ship only if it cuts
  the live hold AND stays smooth. **Fallbacks if both fail:** INT8 quant (INT8-conv kernels DO exist on sm_120 —
  the honest FP8 redo, quality gamble); a turn-start "about-to-speak" gesture to MASK the lag (low risk,
  reuses `MUSETALK_IDLE_MOTION`); `MUSETALK_SYNC_MODE=live` (voice always first, lips trail ~0.75s — direct
  symptom cure, previously rejected); or the structural fix, a **dedicated avatar GPU** (the only guaranteed
  one; avatar working set ~5GB). Full record: `project-visualllm-fp8-quantization-deadend` memory + STATUS.md
  3rd-session block.

---

## P21 — LLM cloud hop was the dominant TTFO cost + all its variance ✅ IMPROVED (2026-07-03, pin OpenRouter to Groq)

**Symptom.** TTFO occasionally spiked to ~7–8s (both languages) for no visible reason; even typical turns
carried a 1–2.5s "nothing happening" gap before the voice. P19 left this as the deferred open item ("LLM
cloud-hop variance 0.8–7.3s now dominates worst-case").

**Diagnosis (measured, `scripts.measure`, 5×/lang, steady).** The full TTFO budget, median:
- **en:** LLM 1.07s + TTS 1.88s + steady-hold 0.78s = ~3.95s
- **zh:** LLM 1.64s + TTS 1.72s + steady-hold 2.03s = ~5.16s

STT folds to ~0 (Deepgram final is ready when the VAD declares stop). The LLM is **pre-warmed on connect**, so
the measured LLM hop is *pure cloud TTFB* — no cold start. It was the single largest component **and** its
whole variance/tail: same Gemini model gave 0.63s, 1.06s, 1.40s, 3.52s in one session — a transpacific
`OpenRouter → Google Gemini` round-trip from Thailand. VAD `stop_secs=0.5` is *before* the TTFO clock
(`metrics.py` starts at `UserStoppedSpeaking`), so it's perceived latency, not in the number.

**Fix — pin OpenRouter to Groq's backend (config-shaped, ~5 lines, no new key).** `build_chat_completion_params`
ends with `params.update(settings.extra)`, and the OpenAI SDK forwards an `extra_body` kwarg into the request
JSON — exactly where OpenRouter reads its `provider` hint. So:
- `pipeline/config.py`: new `openrouter_provider_only` field (env **`OPENROUTER_PROVIDER_ONLY`**).
- `pipeline/stages/llm.py`: if set, `Settings(model=…, extra={"extra_body": {"provider": {"only": [...]}}})`.
  Empty = today's unpinned Gemini (safe default).
- `.env`: `OPENROUTER_MODEL=meta-llama/llama-4-scout` + `OPENROUTER_PROVIDER_ONLY=Groq`.
Reuses the existing OpenRouter key; fully revertible. (Verify the model is Groq-served on OpenRouter first —
Qwen-2.5-72b 404s on Groq. `OPENROUTER_PROVIDER_ONLY` is NOT in the config-panel `_KNOWN` set, so edit `.env`
directly, not via /save.)

**Result (measured).** LLM hop **halved + tail killed**: zh 1.64→**0.80s** median (max 3.59→1.44); en
1.07→**0.67s** (max 2.43→1.35). Isolated probe: Groq TTFT ~0.67–0.91s tight vs Gemini 1.1–1.6s + 3.6s tail.
zh Traditional-Chinese quality holds (natural; one rare simplified-char slip, inaudible via CosyVoice). **HONEST:
end-to-end TTFO only modestly better** — en ~3.95→~3.5s median, zh median ~flat (mean 5.33→4.79s, worst
6.58→5.25s) — because the LLM was only ~1/3(en)/~1/6(zh) of the budget; the shared-GPU **TTS (~1.8s) +
steady-hold (~2.0s zh) now dominate** and are noisy. The real win is variance/tail: Groq TTFT is bounded, so a
multi-second LLM spike can no longer happen.

**Model selection — baseline = `meta-llama/llama-4-scout` (2026-07-04, after a wider search).** It beats the
first pick (`llama-3.3-70b`) on every axis: same speed (Groq, non-reasoning, TTFT ~0.6–1.1s), clean Traditional
zh that is *substantive + accurate*, and **~5× cheaper**. **Pricing correction:** pinning `llama-3.3-70b` to
Groq actually costs **$0.59/$0.79 per 1M**, not the "$0.10/$0.32" first quoted — OpenRouter's `/models` returns
the *cheapest* provider (DeepInfra), while `/endpoints` gives the per-provider price you actually pay when you
pin. scout-on-Groq is $0.11/$0.34. In a clean 5-question isolated eval (same system prompt): scout gave correct
Taiwan-idiomatic answers (台北101 w/ specifics, a real boiled-egg recipe); `llama-3.1-8b` had ERRORS (認主意
nonsense, mislabeled 四四南村 as a night market, truncated answers) → rejected; `llama-3.3-70b` good but terser +
the ≤10-char first-sentence rule made it emit broken openers ("人工智慧是。"). Every *mid-cost* model is a
REASONING model (gpt-oss-20b/120b, qwen3-32b) → slower TTFT; Qwen2.5-72b (non-reasoning, Chinese-native) is only
on non-fast providers (DeepInfra/Novita, ~1.2–1.4s) + leans mainland vocab (計算機/信息). **The fast +
non-reasoning + clean-Traditional set on Groq is just the Llamas** (scout=win, 3.3-70b=good-pricey, 3.1-8b=errors).
**METHOD GOTCHA:** `pipeline.log` never records a turn's OUTPUT text (only prior-turn context + token counts),
and `measure` connects fresh single-turn — so you CANNOT judge a model's live reply quality from the log; use an
ISOLATED multi-question probe (an early "8b looks shallow live" read was stale log data). Token *streaming* is
identical across scout/70b/8b (word-sized deltas, sub-10ms inter-token gaps, no bursting) — not a differentiator.
Scout's English style runs longer + tends to end every reply with a follow-up question (fine in zh; a possible
en tweak via the English system prompt, deferred).

**Stacked lever — zh short-first-sentence prompt.** CosyVoice prefills the whole first sentence before any
audio, and the first-clause split (`COSYVOICE_FIRST_PIECE`, the en lever) barely fires for zh. Adding
"第一句話要特別短（十個字以內），先講重點" to the mandarin system prompt (`config.py`) trimmed the zh TTS
first-chunk: zh TTS hop 1.82→1.67s, zh TTFO ~5.14→~4.34s median (~0.3–0.5s), quality intact. (Llama occasionally
ignores the ≤10-char rule on definitional questions.)

**Goal tightened `<8s` → `<3s`** the same session (docs + `TTFO_TARGET_SECONDS` default + `TtfoMeter` +
`measure.py` display). Current en ~3.5s / zh ~4.3s now sit **over** the bar → every turn logs `[TTFO OVER]`.
The remaining ~0.5–1.5s is shared-GPU-bound; the realistic path to 3s on zh is a **dedicated avatar GPU** (frees
TTS + render at once). `MUSETALK_SYNC_MODE=live` would erase the steady-hold but is **rejected by the user** —
the voice leads the lips ~1–2s (keep `steady`). NOT git-committed (held for live A/V sign-off). Memories:
`project-visualllm-llm-groq-pin-ttfo`, `feedback-visualllm-steady-not-live`.

## P22 — zh steady-hold excess: `COSYVOICE_FIRST_HOP_ZH` reverted 5→0 ✅ FIXED (2026-07-04, live A/B; resolves P19's caveat)

**Symptom.** zh TTFO median ~4.1s vs en ~3.0s on the same stack. Handoff decomposition (`scripts.measure`,
fresh-`[TTFO]`-guarded runs): LLM (~0.7s) and TTS first-chunk arrival (~2.1–2.5s) were ~IDENTICAL across
languages — **the whole zh excess was the steady-hold** (en ~0.85s vs zh ~1.9–2.2s).

**Root cause.** Exactly P19's warning, now measured in isolation: `COSYVOICE_FIRST_HOP_ZH=5` makes CosyVoice
emit a *small* opening audio chunk, which fills the `MUSETALK_LEAD_FRAMES=14` synced-start cushion slowly —
the pump holds the voice until 14 frames are queued, so a small first chunk = a long hold. The hop's isolated
first-chunk TTFB win never survives the live synced-start fill.

**Fix.** `COSYVOICE_FIRST_HOP_ZH` default flipped `:-5` → `:-0` in the cosyvoice repo's `run_vllm_server.sh`
(hop0/lead14 was already eye-validated smooth in P19). **A/B (4 runs/arm, restart dance with the VRAM load
order):** zh median 4.14 → **3.09s** (hold → ~0.8s); smoothness screen clean — freeze ≤147ms, delivered-audio
gaps ≤63ms, fps ≥12.0, holds drain monotonically, lips-start *improved* +0.6–0.9 → +0.3–0.4s.

**Candidate `MUSETALK_LEAD_FRAMES=8` — REJECTED by the live eye (2026-07-04, later the same day).** With
hop=0 it measured **zh 3.03 / en 2.48s median — the first all-runs-under-3s config** (hold ~0.5s), screen-clean
on the probe, and one live probe turn at lead=8 hit 2.50s. But the user then live-tested **every lead value
below 14** and saw delay or avatar freezes on all of them — the P19 lesson repeated: the probe screen (fps,
freeze-ms, audio gaps) does not capture what the eye catches. `.env` stays `lead=14`; **the lead question is
CLOSED — do not re-try lower leads.** Further TTFO gains must come from the TTS first-chunk / P20 stagger /
dedicated GPU instead. (lead=10 skipped — P19's data shows the synced-start delay cliff sits between 10 and 8,
no gain at 10.) en control at lead=14 was noisy (3.28/4.73/5.64 — the P20 turn-start GPU collision,
hop/lead-independent), so treat the en lead8-vs-14 delta as directional. Ops gotcha from the trial: `LEAD_FRAMES`
reaches the avatar server only via a full relaunch (launcher / `run.ps1`) — the config panel's Restart cycles
the pipeline only, so panel-driven lead changes silently never take effect server-side.

**Ops note (cost an hour).** Launch the WSL cosy server from a shell whose wrapper survives:
`wsl bash -c "nohup bash run_vllm_server.sh > /tmp/cosy.log 2>&1 &"` from a normal call gets SIGTERMed with
its wrapper when the calling shell exits — verify the uvicorn process is reparented to `/init` (PPID 1) and
`/health` answers before moving on. The in-place kill/restart keeps the WSL IP (no `wsl --shutdown`!).

## P23 — zh long-opener TTS TTFB ~3.1s: comma-only first-piece split ✅ FIXED (2026-07-04, `COSYVOICE_FIRST_PIECE_ZH=1`)

**Symptom.** After P22, live zh still had a ~4.6–4.8s tail: whenever the LLM ignores the ≤10-char-opener
prompt rule (~30% of turns, mostly definitional questions) and writes a 19–44-char first sentence, CosyVoice
prefills it whole → TTS first-chunk ~3.1s → the turn lands ~4.8s.

**Root cause.** The existing first-clause split (`COSYVOICE_FIRST_PIECE`, the en TTFO lever) is a near-no-op
for zh: it keys on the ASCII comma and spaces, but zh clauses end at the FULL-WIDTH comma ，and have no
spaces — so the en triggers never fire on Chinese. (The earlier "don't split zh" verdict was about *char-cap*
splits cutting mid-word — 天氣預|報; a comma-only split cannot do that.)

**Fix (`local_services/first_piece_aggregator.py`, +47 lines, env-gated, code default 0).** Flush the turn's
FIRST piece at a full-width ，；： only — never a char cap. `COSYVOICE_FIRST_PIECE_ZH_MIN_CHARS=5` guard,
counted in CJK chars (ASCII like "AI" excluded → conservative): the opening piece's audio must cover the next
piece's synthesis or the voice pauses between clauses (zh speaks ~4–5 chars/s, 5+ chars ≈ 1.2s+ audio, and
steady's lead cushion adds slack). A ， before the min → wait for the next boundary; a comma-less sentence →
byte-identical to before. Only the first piece of a turn; `handle_interruption`/`flush` re-arm it. The knobs
are read via `os.getenv` inside the aggregator so the `cosyvoice_tts.py` call site stays untouched. The
trailing ， is kept on the emitted piece (CosyVoice's zh normalizer uses it for clause prosody; the en path
strips its ASCII comma — unchanged).

**Verification ladder (all pass).** (1) Offline unit: comma@4<min→next comma; comma-less unsplit; en pieces
byte-identical knob on/off; later sentences whole. (2) `scripts.preflight`. (3) Isolated `/tts/stream` TTFB,
read(1024) small-buffer (a 64KB read masks TTFB): 22-char full **3.07s** vs its 10-char comma piece **1.88s**.
(4) Live A/B (hop0/lead14, fresh-TTFO): long-opener turns **4.78→3.08s / 4.83→3.42s** (TTS span −1.2–1.5s);
**split-fired audio gaps 59–65ms max** — no between-clause pause, MIN=5 sufficed; comma-less + en buckets
unchanged. Also fires at full-width ：(by design; observed live, smooth).

**Test-tooling notes.** Deepgram garbles CosyVoice-synthesized question wavs (and the ElevenLabs key is dead,
401) → `edge-tts` (user pip, test-only) synthesizes the zh question wavs (`output/_zh_q_def/why/wx.wav`).
Synthetic-mic runs can be interruption-contaminated: the VAD splits the wav's internal pause into two user
turns → the first reply is cancelled mid-stream; filter those by `llm→tts > 1.0s`. Weather questions
("台北明天的天氣…請詳細…") reliably elicit the long comma-opener shape for A/B.

## P24 — Phone browser plays the voice through the EARPIECE ⏳ SHIPPED v2, awaiting phone re-test (2026-07-04, `CLIENT_FORCE_SPEAKER=1`)

**Symptom (user report).** On a phone browser the avatar's voice comes out of the quiet earpiece (hold-to-ear
phone-call routing), not the loudspeaker.

**Root cause.** When a WebRTC page holds a live mic, Android Chrome flips the phone into 'communication'
audio routing → remote audio goes to the earpiece. (iOS has its own variant: no output-selection API at all.)

**Fix (`pipeline/main.py::_install_client_speaker_route()`, the sanctioned env-gated `<head>` injection —
bundle untouched).** v1 (DOM sweep + `setSinkId`) FAILED the user's live test; three real gaps, all fixed in v2:
1. **Cache:** the phone can serve the pre-patch index from cache and silently miss every injected fix → the
   patched index is now served `Cache-Control: no-store`.
2. **Invisible elements:** the bundle can play audio through an element never attached to the DOM (or in a
   shadow root) — a `querySelectorAll` sweep misses it. v2 hooks `HTMLMediaElement.prototype.play` so every
   element that ever plays gets routed.
3. **iOS:** no `setSinkId`/output labels → v2 falls back to piping the element's MediaStream through an
   AudioContext (its output uses the media/playback route = loudspeaker, not the 'communication' route). The
   element is muted only AFTER the context is confirmed running (audio can never vanish); a one-shot
   pointerdown resume covers iOS's autoplay/gesture rule.
Mobile user-agents ONLY — a desktop user on headphones must never have audio yanked to the speakers.

**Diagnosability (the real lesson).** The injected script POSTs each step (loaded / devices found / which
route fired / errors) to `/client/speaker-debug`, logged as `[speaker-debug]` in `pipeline.log` — a remote
phone becomes debuggable from the log instead of guesswork. GOTCHA that ate the first beacon test: pipecat's
runner `logger.remove()`s every sink at startup and the file sink only returns when a bot session starts
(`log_setup.ensure_file_sink`) — pre-connect beacons vanished silently (the handler returned 204 but logged
nothing). The beacon handler now re-asserts the sink itself. Also structural: the jitter-buffer + speaker
patches register into ONE shared middleware/patch list — two separate index-serving middlewares would shadow
each other (last-added runs outermost and wins).

**Status.** Both injections verified served on `/client/`, JS `node --check`ed, beacon round-trip verified
into `pipeline.log`, full measure run healthy after the change. Awaiting the user's phone re-test (fully
close the tab first — though no-store makes staleness a one-time issue). `CLIENT_FORCE_SPEAKER=0` disables.

## P25 — The two P20 turn-start levers (GPU stream priority + CosyVoice stagger) ⚠️ BOTH REJECTED, measured (2026-07-05); and the P20 premise is FALSE on the TRT baseline

Implemented and live-A/B'd both untried P20 levers. **Both fail, and the measurement overturns P20's root-cause
premise.** The knobs are kept default-OFF (inert, byte-identical to baseline) but marked measured-dead.

**Baseline first (zh, steady, lead=14, TRT+GPU_COMPOSITE, fresh `[TTFO]` per run):**

| turn | TTFO | LLM hop | TTS first-chunk | avatar **hold** |
|---|---|---|---|---|
| _zh_q_def | 3.09 / 2.91 / 2.12s | 0.58 | ~1.5–2.1s (dominant) | **~0.45s (rock-steady)** |
| _zh_q_why | 2.95s | 1.02 | ~1.5s | ~0.43s |
| _zh_q_wx | 2.38s | 0.63 | ~1.3s | ~0.46s |

**Lever 1 — high-priority CUDA render stream (`MUSETALK_HP_STREAM=1`, `musetalk_server/app.py`
`_init_hp_stream` + `render_segment` wrapper). CATASTROPHIC live: TTFO ~12s** (hold ~10s) on every turn; TTS
unaffected — pure render collapse to ~1–2fps. **Isolating test nails the cause:** driven OFFLINE (no CosyVoice,
`_drive_frames … burst`, `MUSETALK_PROFILE=1`) the HP stream renders **identically** to baseline
(gpu ~101ms vs ~103ms per 8-frame seg, ~14fps both). So the priority stream is harmless *within one process* —
it only self-destructs **live, contending with CosyVoice**. Mechanism: CUDA stream priority orders work only
**within a CUDA context**; MuseTalk (Windows) and CosyVoice (WSL2 VM) are two processes under the **WDDM**
scheduler, which arbitrates the shared card. Demanding to preempt another process/VM's kernels doesn't get
served first — it makes WDDM **thrash** the GPU between the two contexts (expensive cross-process/VM context
switches on consumer Blackwell), chopping the render into overhead-dominated slices. This is worse than the
handoff's predicted "no-op" — it's actively harmful. Cross-process priority would need **CUDA MPS (Linux-only)**;
not available on Windows+WSL. **Dead on this topology.**

**Lever 3a — CosyVoice turn-start throttle (`COSYVOICE_PACE_RATE`, cosy repo `app.py` `gen()`; the knob the
docs cited never actually existed — added it: pace the opening chunks to `rate`× real-time for the first
`COSYVOICE_PACE_WINDOW_S` of audio, never delaying the first chunk). USELESS + UNSTABLE at `rate=1.5`:** it did
**not** reduce the hold (def 0.48 ≈ baseline 0.47; wx 0.43 = baseline) and instead made it **wildly variable** —
_zh_q_why hold swung **0.39 → 2.06 → 5.33s** across repeats (TTFO up to 7.17s). Throttling CosyVoice's feed just
**starves the avatar of audio to render**, so in steady the voice is held waiting for frames. No upside, large
unpredictable downside.

**The real finding (corrects P20):** on the **TRT+GPU_COMPOSITE baseline the turn-start render is NOT
GPU-starved.** Offline profile ~101ms/8-frame-seg (≈78fps raw), and the live hold is a steady ~0.45s = the
**structural `lead=14` synced-start fill**, not contention. Both levers targeted a collision that isn't the
bottleneck here, so neither could help and both introduced instability. The dominant, *variable* TTFO cost is
the **TTS first-chunk (CosyVoice prefill, ~1.3–2.1s, scales with the first sentence's length)** — untouched by
either lever (both protect the first chunk). Corollary: a **dedicated avatar GPU won't cut turn-start TTFO
either** (the hold is lead-fill, not contention); it only helps LONG-reply drift (P16, which *is* contention).
**The only remaining TTFO lever is shrinking the TTS first-chunk** (already partly done: the en/zh first-piece
splits P23 + the short-first-sentence prompt P21) or the structural `lead` (eye-rejected below 14, P22).
`MUSETALK_HP_STREAM` / `COSYVOICE_PACE_RATE` stay in-tree, default 0, documented dead.

## P26 — Filler("thinking")-word opener ✅ SHIPPED as the new baseline (2026-07-05, `FILLER_WORDS=1`)

Given P25 (the only real TTFO lever left is the TTS first-chunk), added a **filler-word opener**: the turn starts
on a short, canned "thinking" phrase that synthesizes fast, so the bot starts speaking before the slow real
sentence is ready — a human "umm, let me think…". `local_services/first_piece_aggregator.py` (extends the
existing `FirstClauseAggregator`, so it needs `COSYVOICE_FIRST_PIECE=1`); env-gated, default OFF in code, `.env`=1.

**Design — SAFE by construction.** The filler is emitted as a normal TEXT piece through the existing TTS path,
NOT injected as raw audio. So it rides the proven per-turn framing (`push_start/stop_frames` → one
`TTSStarted/Stopped` per turn → the avatar's `speech_start/end`): the filler + real reply are ONE continuous
turn, no screech/desync risk (verified: delivered audio gap ~60ms). At the turn's first LLM token the aggregator
yields `FILLER_WORDS_COUNT` fillers from a rotated pool (no immediate repeat) before the real content.

**Result (zh, vs baseline).** def 2.91→**2.23s**, wx 2.38→**2.03s**, why ~flat (its filler + a slow-LLM run);
holds stayed steady ~0.49s. **~0.7s sooner to first sound.**

**How TTFO counts this — the honest caveat (the user asked directly).** `TtfoMeter` = `UserStoppedSpeaking` →
`BotStartedSpeaking` = **time to first SOUND out**, content-agnostic. With the filler on, the first sound is the
filler, so the 2.23 is time-to-first-*filler*. The **real answer arrives slightly LATER** (queued behind the
filler): e.g. def real-content ~2.91→~3.4s. So this is a **perception/responsiveness win, not a speedup** — the
avatar *starts responding* sooner (mouth moves, voice on) but the *answer* isn't faster. Measuring "time to first
real answer" would show a small negative. Shipped because for a companion avatar, "feels alive sooner" is the
goal; recorded so future TTFO numbers are read correctly (they now include the filler head-start).

**Tuning learned (the wx regression + fix).** First cut used one-syllable fillers ("嗯，", ~0.3s). In steady the
pump holds the voice until `MUSETALK_LEAD_FRAMES` (=14, ~1s audio) render — a 0.3s filler can't fill that cushion,
so the **hold ballooned 0.5→1.7s** (wx 2.94s, a regression). Fix = a pool of **~1.2s thinking phrases**
("嗯，讓我想一下喔，") so the first piece alone fills the cushion → holds back to ~0.49s. Tension: longer filler
fills the cushion but its own first-chunk prefill is slower (why the longest-filler run stayed flat) — sweet spot
≈ the lead-cushion length. `FILLER_WORDS_COUNT` chains more (each adds ~1.2s of "thinking" before the answer).

**Open follow-ups (not done):** fire the filler only on slow/long-opener turns (it currently fires every turn,
incl. trivial "你好" replies — the main taste risk); add a "first real answer" metric next to TTFO; or the
instant **cached-audio** variant (play a pre-recorded filler on `UserStoppedSpeaking`, before the LLM — hits ~1s
since it skips the LLM-first-token + synth gates, but needs the riskier raw-audio injection this version avoids).

## P27 — TTS first-chunk cut ~2.0→~0.85s via vLLM CUDA graphs + tighter stream poll (levers "4+2", 2026-07-05, 7th session)

**Opportunity.** After P25/P26, STATUS called the **TTS first-chunk** (CosyVoice prefill, ~1.3–2.1s, scales with
opener length) "the ONLY remaining TTFO lever." Every prior attempt shrank the *input* (first-piece splits, hop —
and hop's isolated win kept dying live, P19/P22). These two levers instead cut the *compute* so the same first
chunk **arrives sooner** — a fundamentally different, safer mechanism.

**The two levers** (both in the cosyvoice repos; uncommitted, per the established hold-for-live-A/V pattern):
1. **CUDA graphs (Lever 2, dominant).** `run_vllm_server.sh` now defaults `COSYVOICE_VLLM_EAGER=0` (was 1). vLLM's
   speech-token LLM ran eager (no CUDA-graph capture — skipped originally as "needs more toolchain" on the
   Blackwell/WSL stack). Non-eager captures graphs → kills per-token kernel-launch overhead on the ~28-token
   first-chunk decode.
2. **Poll tighten (Lever 4, small clean add).** `CosyVoice/cosyvoice/cli/model.py` streaming loop
   `time.sleep(0.1)`→`0.02`. The producer/consumer polled ready-tokens at 100ms granularity, wasting up to
   ~100ms at the first chunk (~0.04–0.08s of the win).

**Measurement — two independent instruments agree** (the confidence bar):
- *Isolated TTS TTFB* (controlled exact text, median-of-5, hitting the **WSL IP directly** to bypass the
  localhost-relay stream buffer): en 1.05→**0.75** / 2.03→**1.26**s; zh 1.55→**1.05** / 2.36→**1.42**s — **−30–40%.**
- *Live end-to-end TTFO* (`scripts/measure.py`, real turns): en 1.91–2.2s (median ~2.0, TTS chunk ~0.85s);
  zh 1.98–2.56s (median ~2.25, TTS chunk ~0.9s) — **all PASS <3s** (vs memory baselines en ~3.5 / zh ~3.36s).
  The live TTS-chunk (~0.85s) maps straight onto the isolated probe → the win transfers.

**Why it survives live where `hop` didn't.** hop made the first chunk *smaller* → filled the
`MUSETALK_LEAD_FRAMES=14` cushion slower → its isolated win evaporated. CUDA graphs makes the *same-size* chunk
*arrive sooner* → pure upside for the synced start (the pump gets its lead frames earlier). The P19/P22 erosion
mechanism structurally does not apply here.

**Costs / caveats.** (1) Slower boot: graph capture + inductor compile adds ~30–60s (one-time per launch; the
launcher's `/health` wait tolerates it — verified). (2) One-time Triton JIT spikes on novel sequence shapes until
cached (warmup absorbs the main one — watch the very first turns after a fresh launch). (3) VRAM unchanged
(10.3GB; capture cost ~0 at `gpu_util 0.16`) → no load-order-risk change. **Revert** = `COSYVOICE_VLLM_EAGER=1` +
relaunch (poll edit is inert on its own).

**Verified:** server boots clean with `enforce_eager=False`, no OOM; 20+ syntheses across both A/B probes + 7 live
turns (3 en, 4 zh) all PASS; both A/B runs under matched conditions (MuseTalk + pipeline up, GPU mem identical).
The honest read: much of the *perceived* wait was already masked by `FILLER_WORDS`/`FIRST_PIECE` (P23/P26), so
this shows up as a real **compute** win (freed GPU, lower TTS-chunk) more than a dramatic stopwatch drop — but it
directly shrinks the one cost STATUS had flagged as the wall. Follow-up: Lever 1 (flow-matching TRT) was tested and
**REJECTED** — see P28.

## P28 — flow-matching TensorRT (Lever 1): built, tested, REJECTED (no TTFO win, worse throughput) (2026-07-05)

**Hypothesis (from P27).** After CUDA graphs, the remaining TTS first-chunk compute is the flow-matching estimator
(10 Euler steps x CFG batch-2 = 20 fp32 passes/chunk, PyTorch). CosyVoice ships `flow.decoder.estimator.fp32.onnx`
with a dormant `load_trt=False` path; run it on TensorRT fp16 (~1.9x, same pattern as the MuseTalk TRT win, P16).

**What was done.**
- **TensorRT installed** in the WSL `cosyvllm` env: `pip install tensorrt` → `tensorrt_cu13==11.1.0.106` (TRT **11**,
  CUDA 13). Dry-run first proved it **purely additive** (4 new `tensorrt_cu13*` pkgs, no existing-pkg changes) → the
  vLLM baseline was never at risk. Env snapshot `cosyvoice-local-tts/logs/_env_snapshot_pre_flowtrt.txt`.
- **Ported CosyVoice's TRT-8/9 build helper to TRT 11** (`CosyVoice/cosyvoice/utils/file_utils.py`): 3 patches —
  `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` removed (→ `network_flags=0`); `BuilderFlag.FP16` removed (TRT 11 is
  strongly-typed only — fp16 needs an fp16 ONNX, not a flag) → decoupled fp16 from `COSYVOICE_FLOW_TRT`; network I/O
  `.dtype` now read-only (→ try/except). The runtime API (`execute_async_v3`, `set_tensor_address`, …) is unchanged
  on TRT 11. The **fp32 engine builds + runs clean** (289MB `.plan`, verified I/O order correct via TRT introspection).

**Method lesson (cost ~an hour of false "it's broken").** The first synth through the engine crashed (tiny 2-frame
mel → vocoder "kernel > input"; + a "Float vs BFloat16" error) — looked like a TRT bug. It was **my test harness**:
the build/validate script set `COSYVOICE_VLLM=0` to save VRAM, which ran the speech-token LLM in **PyTorch** mode (a
different path than prod's vLLM) → bf16/fp32 mismatch starved the flow. In the real config (vLLM ON + flow-TRT) it
runs fine. **Always validate a TRT/flow change in the SAME config as prod (vLLM on), never the `load_vllm=0` shortcut.**

**Results — rejected on BOTH measurable axes:**
- *TTFO / first-chunk (isolated TTFB, WSL-IP direct):* **flat vs P27** (en 0.75→0.79/1.26→1.25, zh 1.05→1.08/
  1.42→1.36 — within noise). The flow is NOT the first-chunk bottleneck: the first chunk is ~25 tokens ≈ one short
  flow pass, dominated by the vLLM prefill+decode that P27's CUDA graphs already optimized.
- *Throughput (controlled 10-run RTF A/B, fixed long text):* **~26-40% SLOWER**, not faster — RTF en 0.314→0.395,
  zh 0.306→0.429 (total synth ~4.9s → 6.1-6.7s for ~16s of audio). Root cause: `flow_matching.forward_estimator`'s
  TRT path calls `torch.cuda.current_stream().synchronize()` **before and after every estimator execute** (~20 hard
  GPU stalls/chunk) → serialization swamps any fp32 fusion gain (PyTorch runs the passes async). NOTE: an earlier
  single-sample probe *hinted* ~15-20% faster — that was NOISE (unequal stochastic audio lengths); the RTF-normalized
  A/B reversed it. Measure throughput with fixed text + RTF, never raw totals across variable-length samples.

**Verdict.** flow-TRT is useless (fp32: net-negative) for this stack — wrong tool for TTFO, and a throughput loss.
fp16 (~1.9x on flow *compute*) would have to overcome the same per-pass sync overhead AND needs an fp16-ONNX
conversion (onnxconverter_common + op blocklist, accuracy risk) for a payoff bounded by the flow's small first-chunk
share → not worth it. Reverted to the P27 baseline. Everything left in-tree is **env-gated `COSYVOICE_FLOW_TRT`,
default OFF, inert** (the 3 TRT-11 build patches, `tts_engine.py` wiring, the fp32 `.plan`, `_build_flow_trt.py`).
If TTS first-chunk ever needs more, the lever is the LLM decode (already CUDA-graphed) or a distilled 1-NFE
token2mel (IntMeanFlow), NOT flow-TRT.

## P29 — Local sherpa STT integrated; `[TTFO]` count:0 + mic-stranded-after-greeting FIXED (2026-07-05, 8th session)

**Context.** Brought the local OFFLINE STT (`STT_PROVIDER=sherpa`) from `feat/offline-stt-sensevoice` onto the
current `feat/ttfo-first-clause-split` stack — surgical port (STT-only hunks; a full merge would regress the newer
TTFO work). sherpa-onnx streaming zipformer, bilingual zh-en, in-process, CPU/~0 VRAM, zh→Traditional via OpenCC,
drives turns from its own ASR endpoint. funasr (SenseVoice `:8004`) ported as an untested alt. Deepgram stays
default. Commits `5bacfa8` + `df278fd`.

**Bug 1 — `[TTFO]` meter read count:0 on the ASR path.** `TtfoMeter` armed only on `UserStoppedSpeakingFrame`, but
sherpa drives turns with `VADUserStoppedSpeakingFrame` — a `SystemFrame`, NOT a subclass (verified via MRO). Fix:
arm on either (`pipeline/metrics.py`); Deepgram unchanged (both frames mark the same instant).

**Bug 2 (the real blocker) — mic stranded after the greeting under steady.** sherpa paused decoding while the bot
spoke (echo suppression) and resumed on `BotStoppedSpeakingFrame`. But under steady the screech fix pins
`BOT_VAD_STOP_FALLBACK_SECS=600`, so that frame never fires (**same mechanism as P11's broken echo-guard mute**):
after the connect greeting `_bot_speaking` stayed True forever and every later mic frame was dropped → no
transcript, no turn, meter never armed. Found by instrumenting `run_stt` (1651 consecutive PAUSED calls, no
`BotStopped`). Fix: **gate the pause behind echo-guard** (`pause_while_bot_speaks=cfg.echo_guard`) — echo-guard is
default OFF and only valid with live sync (where BotStopped fires), so under steady the mic stays live (the
documented barge-in/headphones tradeoff) and the pause can't strand. Verified via the WebRTC probe: sherpa zh turn
→ LLM → CosyVoice → avatar, `[TTFO] {count:1, median ~2.1s, pass:True}`. Also declared `STTSettings(model/language)`
to silence the harmless NOT_GIVEN validation error. Real-mic confirmation is still the user's to do (use headphones).

## P30 — Chinese "avatar first, voice delayed" = the FILLER opener colliding with zh's short pieces (2026-07-05)

**Symptom.** With `FILLER_WORDS=1`, zh turns felt like the avatar started, then the voice lagged. **en fine.**

**NOT the cause: raw zh TTS speed.** This session zh TTFB = 0.72–1.07s, en = 0.71–1.49s — equal (the old ~2.3s zh
first-chunk penalty was gone). So the fundamental zh-slow-chunk (P15) is no longer the issue.

**Root cause.** zh replies are chopped into very short pieces (filler + sentence/comma splits), e.g.
`[欸，說到這個問題，]` → `[你好！]` → `[有什麼需要幫忙的嗎？]`, each a separate TTS call with ~0.8s TTFB. Each
short piece's audio (~0.6–1s) barely covers the **next** piece's synth, so in steady mode (video paced to voice)
the voice micro-pauses between pieces. The filler adds one more short seam at the front AND pushes the real answer
later → accumulated gaps = "avatar first, voice delayed." en escapes it: en fillers are long
("Well, let me think about that, ") and en splits by char-count (MIN 18 / MAX 32) → bigger pieces that each cover
the next synth. **Fix:** `FILLER_WORDS=0` (confirmed smooth in zh; costs ~0.7s TTFO) or make the filler en-only.

## P31 — CUDA graphs (P27) caused live per-turn INCONSISTENCY → reverted to eager (2026-07-05, 8th session)

**Symptom.** After P27 (`COSYVOICE_VLLM_EAGER=0`, CUDA-graph capture), the live system was "very inconsistent"
per turn — some turns fast, some spiking. User-confirmed root cause: reverting to eager fixed it.

**Why.** A captured CUDA graph is **shape-specific** (a token/seq-length bucket). Real conversation = variable
reply lengths, so turns keep hitting shapes not yet captured → one-time capture / Triton-JIT spike on that turn
(fast, fast, SPIKE, fast…) = the variance. Worse on the shared 16GB card: rigid graph replay can't flex around
MuseTalk's concurrent TRT render, whereas eager issues kernels dynamically so the scheduler interleaves them. Graphs
optimize *average* TTFB (−30–40%) by trading away *consistency* — right for fixed-shape batch serving, wrong for
variable-length realtime turns on a contended GPU.

**Fix.** `run_vllm_server.sh:24` default flipped back to `COSYVOICE_VLLM_EAGER=${...:-1}` (eager). Slower first-chunk
baseline (~2.0s vs ~0.85s) but STEADY — for a live talking avatar, consistency beats fast-on-average. The
independent Lever-4 poll-tighten (`model.py` 0.1→0.02) stays. Graphs would only pay off with fixed-length replies
or a dedicated TTS GPU. (Revises the P27 "new baseline" verdict.)

## P32 — CUDA graphs re-investigated (diagnose-then-fix): the two P31 mechanisms did NOT reproduce on the TTS side (2026-07-05, later 8th session)

**Ask.** "Can we use graphs without the drawback?" Built an isolated first-chunk **TTS-TTFB variance** probe
(`cosyvoice-local-tts/_ttfb_variance.py`: median/**max/stddev**, varied opener lengths × N rounds, hits the WSL IP
directly, first-sighting-vs-repeat to expose capture spikes).

**Findings — both P31 mechanisms fail to reproduce in the TTS layer.** (A) *Cold-shape capture:* a FRESH graphs
server (only the 1-shape boot warmup) showed **no** round-1 penalty (−0.03s), no spike flags — vLLM V1 doesn't
re-capture per reply length. (B) *Contention:* under a continuous GPU hog AND real MuseTalk TRT render, graphs are
**faster AND lower-variance** than eager (96 samp under MuseTalk: graphs median 1.29 / max 2.23 / stddev 0.37s vs
eager 1.94 / 3.43 / 0.64) — the *opposite* of "rigid graphs fight MuseTalk." So on the TTS stopwatch graphs look
strictly better, and the band-warmup fix I'd designed was NOT built (mechanism A doesn't exist).

**BUT this measured the wrong side — see P33.** Spec: `docs/superpowers/specs/2026-07-05-cuda-graphs-without-drawback-design.md`.
Lesson: a green TTS-TTFB probe is not a green avatar.

## P33 — VERDICT: keep EAGER — CUDA graphs degrade zh LIPSYNC (the real, measured cost) (2026-07-05, later 8th session)

**Symptom (user's eye).** With graphs ON the zh **mouth shapes stop matching the words** (voice sounds right, lips
wrong); graphs OFF the zh lips match. English fine either way. NOT timing/fps — per-turn render held ~14fps in every
graphs turn (`[avatar timing]` logs).

**Root cause, MEASURED (`cosyvoice-local-tts/_zh_audio_ab.py`, same 37-char zh sentence ×5 per mode).** Graphs ON
alter the zh **audio**: duration median 8.92 vs 8.28s, longest internal silence 0.76 vs 0.68s, silence-frac 0.30–0.36
vs 0.22–0.33, more run-variance. The graph-captured decode perturbs the **zh-critical RAS sampling** (P18 — RAS is
what stops zh looping on the silence token). MuseTalk lip-syncs off a **Whisper of the actual waveform**, not the
text — so a degraded/altered zh waveform → mouth shapes that don't track the phonemes the user hears. en doesn't lean
on RAS, so it's spared.

**Fix / verdict.** Keep `COSYVOICE_VLLM_EAGER=1` (eager). Graphs win the TTS stopwatch (P32) but lose the avatar,
and the avatar is the product. This **reconciles with P31's original revert** — the eye was right; the TTS-side probe
(P32) structurally cannot see the zh-audio cost. Set `=0` only for an en-only / TTS-throughput setup. The config
panel's CUDA-graphs toggle flips + persists this. (Recurring lesson, now 3×: the probe passes what the eye rejects.)

## P34 — zh turn-start "breathing sound" = CosyVoice's leading breath; the trim was REJECTED, no-trim is baseline (2026-07-05, 8th session)

**Symptom (user's ear + eye).** On ~every zh turn a soft **breath/aspiration sounds before the answer**, and the
avatar's **mouth moves along with it** ~0.3–0.6s before the first word. NOT the filler (`FILLER_WORDS=0`, verified
zero filler TTS in the log), NOT the reference clip (`pro_ref.wav` starts clean on 你好, no leading inhale), NOT the
LLM text.

**Root cause, MEASURED (probed the CosyVoice server directly, bypassing the client).** CosyVoice's **zero-shot synth
prepends a low-level breath** before the actual speech — measured 25–610 ms of −34 to −68 dB pre-speech content on
essentially every zh piece (e.g. `想了解更多嗎？` = 610 ms @ −45 dB; `從前，有個勇敢的王子。` = 325 ms @ −49 dB). It's
baked into the returned waveform (the reference prompt is Chinese, so the generation transition out of that
conditioning bleeds a Mandarin breath). Because the first-piece split makes each turn's *opening* its own short
synthesis, the breath lands at the front of every turn. The avatar lip-syncs off a **Whisper of the whole waveform**
(same mechanism as P33), so it renders mouth shapes over the −45 dB breath → mouth moves before the words.

**Attempted fix (REJECTED).** A start-of-turn **leading-breath trim** in `cosyvoice_tts.py`: arm on
`LLMFullResponseStartFrame` (first piece of the turn only), stream-drop the lead until a 5 ms RMS window crosses
−33 dB, keep a 50 ms guard. Offline DSP verified it (avg 256 ms removed, speech attack intact) and **zh TTFO stayed
fine — 2.72 s median (5 turns, 2.11–3.12), no regression** (in `steady` the pump gates `BotStartedSpeaking` on
`MUSETALK_LEAD_FRAMES` fill, not on the first audio sample, so removing the breath doesn't push the start later).
**But it broke live:** `np.frombuffer(lead, int16)` needs an even byte count, and aiohttp streams **odd-sized**
chunks, so `_onset_byte` raised `ValueError: buffer size must be a multiple of element size` on the first piece of
every turn → that piece failed, only the second short sentence (the follow-up question) played = **"cosy only speaks
a sentence each."** The offline test fed fixed 960-byte (even) chunks so it passed — the probe passed what live
rejected (P33's lesson again).

**Verdict.** **No trim — the CosyVoice leading breath is accepted as baseline.** The user judged *without trim is
better* (the breath is minor; the trim added a live failure mode and touches the load-bearing TTS path). The trim was
fully reverted (`cosyvoice_tts.py` back to HEAD, `.env` `COSYVOICE_TRIM_LEAD*` removed). If ever re-attempted, do it
**server-side in CosyVoice** (trim the lead on the raw waveform where buffers are whole, or clean the reference clip),
not as a byte-stream trim in the pipecat client. Diagnosis method kept: probe `:8001` directly and read the
pre-speech dB envelope (`scratchpad_tts/` samples).

## P35 — `[TTFO]` undercounts the real latency the user HEARS by ~1.26s; measure now reports a to-the-ear waterfall ✅ SHIPPED (2026-07-06, 9th session)

**Problem.** The acceptance metric `[TTFO]` (`pipeline/metrics.py`) measures `UserStoppedSpeaking →
BotStartedSpeakingFrame` — i.e. when the pipeline *starts pushing* audio into the transport. Everything after that was
**unmeasured**: transport output pacing, the steady lead-hold, WebRTC encode, network/depacketize, and the browser's
own jitter buffer + decode + speaker playout. The old `scripts/measure.py` compounded the blind spot — its headless
probe recorded frame/packet *arrival* on a separate `time.time()` clock that was never joined to the pipeline log's
clock, so nothing summed to a true end-to-end and the last mile was invisible.

**Fix — a per-stage latency waterfall summing to the user's ear.** Key realization: the probe and the pipeline run on
the **same box = the same wall clock**, so `t0.timestamp()` (log t0), the probe's `time.time()`, and the browser's
`Date.now()` are one epoch. Stitch them → every client arrival is a real offset from t0. Two last-mile sources
(mirroring the existing dual `lip_offset`): **(1) headless, always-on** — the audio pump records `(epoch, rms)` per
frame; `answer_onset_epoch()` finds the first sustained energetic frame after t0 = the answer reaching the client (the
`probe` transport row, no browser); **(2) real browser, opt-in** `CLIENT_PLAYOUT_PROBE=1` (default OFF) — a
`<head>`-injected AnalyserNode taps the bot audio track and beacons `[client-playout] {"ev":"audio-onset","t":<ms>}`
to a new `/client/playout` endpoint (env-gated via the sanctioned `_client_head_patches` injection, bundle untouched);
`scripts.measure --from-browser` parses it into the `browser` row (labeled `browser+net` when there's no headless
probe, since that Δ then absorbs transport). `pipeline/metrics.py` is **deliberately untouched** — the waterfall is
derived in `measure.py`.

**Measured (clean zh turn, full stack):** `[TTFO]`=**2.22s** but the user HEARS the voice at **3.48s** — a **+1.26s
last mile** (transport +0.86s measured, browser +0.40s estimated at the 400ms jitter buffer) that was previously
invisible. LLM +0.68s → TTS first-chunk +1.04s → steady hold +0.50s are the server-side legs.

**Findings the live run exposed (the offline tests couldn't):**
- **Negative (pre-t0) anchor bug ✅ FIXED.** A synthetic-mic turn whose internal pause VAD-split into two sub-turns
  matched an LLM-TTFB log line from *before* this turn's t0 → the LLM row rendered a physically-impossible **−1.34s**
  latency. A stage can't complete before the turn starts, so any negative/`None` anchor now renders **`unknown`**
  (never a fake number); the sum still telescopes through the later real anchors. Locked with a unit test.
- **Live-browser TTFO doesn't log for rapid/overlapping (barge-in) turns** — the TtfoMeter only fires on a clean
  user-stop→bot-start pair; back-to-back turns (bot stop→start 2ms apart) leave it unarmed, so no `[TTFO]`. Pre-existing
  `metrics.py` behavior, surfaced here; not changed.
- **Staleness guard:** `measure.py` reads the LAST `[TTFO]`; if that turn is older than `duration+tail+15s` it blanks
  `client_arrival` with a warning, so a stale t0 can never silently produce a wrong latency.

**Verification state.** Headless path **fully verified live**; browser-beacon path **proven server-side end-to-end via
a synthetic beacon** (script IS served on `/client/`; `/client/playout` 204+logs; `--from-browser` parses it,
`browser+net +0.36s cum 2.50s`) — the ONLY unexercised link is a real browser's AudioContext firing on real voice
(needs a human at the page + mic). 7/7 pure tests green; final opus whole-feature review = Ready-to-merge YES, no
Critical/Important. **10 commits on `feat/ttfo-first-clause-split`, NOT on `main`** (kept on-branch: the browser-beacon
commit hard-depends on held client-patch infra, so "measure only" isn't cleanly separable). Spec/plan:
`docs/superpowers/{specs,plans}/2026-07-06-measure-end-to-end-latency*`. Full run recipe: `CLAUDE.md` Commands →
Avatar A/V test tooling.

## P36 — custom "Nimbus AI" web client (figma-to-code) + the resolution-vs-sync ceiling on the shared GPU ✅ SHIPPED (2026-07-07, 11th session)

**What.** A distinct, self-contained **custom client** at **`/nimbus/`** (a figma-to-code redesign — full-screen
weather-anchor avatar on the left 70%, glass chat panel on the right 30%), served ALONGSIDE the pipecat prebuilt
`/client` (untouched, still the fallback). It speaks the SAME SmallWebRTC signaling — `POST /api/offer`, then avatar
video + bot audio arrive as WebRTC tracks and the mic goes up the same PC — so **plain vanilla JS, no build step, no
pipeline structural change**. Mounted as `StaticFiles` (`_install_nimbus_client`, served `no-store`) from
`local_services/nimbus_client/`.

**Pieces (all additive, all env-nothing / client-only unless noted):**
- **Live A/V core** — validated headless against the running stack: offer→`200`, both tracks arrive, avatar frames
  decode. Connect / **Disconnect** ("End") buttons + a loading→connecting→live overlay (no static poster); a
  name-tag status dot + equalizer that animate only while the bot speaks (AnalyserNode on the bot audio).
- **Text send** — `POST /client/say {text}` injects a TYPED turn into the live pipeline as
  `LLMMessagesAppendFrame(run_llm=True)` (same `_active_task` inject as the measure button) → LLM→TTS→avatar speaks
  it. The box echoes the user bubble locally + auto-connects if not yet live.
- **Transcript bubbles** — the pipeline has no RTVI processor in this build, so instead of a data channel a
  **READ-ONLY `BaseObserver`** on the `PipelineTask` (`observers=[...]`, no structural change) taps the bot's
  aggregated reply (`LLMTextFrame`s bracketed by `LLMFullResponseStart/End`) + finalized user `TranscriptionFrame`s
  into a ring buffer, served by `GET /client/transcript?since=N`; the client polls it (~0.9s) → bubbles. Typed turns
  produce no `TranscriptionFrame`, so no double-add.
- **Avatar = the weather anchor.** `AVATAR_REF=assets/avatar_studio_match.png` — a **square-padded** crop of the
  studio presenter photo. MuseTalk forces a SQUARE output (`_frame_to_bytes` resize to `SIZE×SIZE`), so a wide shot
  would horizontally squish; the padded square keeps the full studio and the panel's `object-fit:cover` crops the
  blurred pad. **Anchored to the TOP** (real image at top, pad at bottom) + `object-position:center top` so the
  studio top is kept and the bottom cropped (user's framing pick). NOTE the model only lip-syncs a **cropped
  256px face** composited back into the full frame — output resolution never sharpens the animated mouth, only the
  static studio/hair.

**LLM 429 dead-end ✅ FIXED (same session).** `OPENROUTER_PROVIDER_ONLY=Groq` pinned the free Groq tier with **no
fallback**, so an intermittent Groq `429 Too Many Requests` killed the whole turn (looked like "no response";
compounded because replies are spoken, not yet text — the transcript above fixes the visibility). Fix = **unpin**
(`OPENROUTER_PROVIDER_ONLY=` empty) → OpenRouter uses its full pool (`deepinfra, groq, novita, google-vertex`) with
auto-failover off a rate-limited provider. Gotcha caught live: the `.env` loader does **NOT strip an inline `#`
comment** from a value — `OPENROUTER_PROVIDER_ONLY=  # note` sent the comment text as the provider list → `404 No
allowed providers`. Keep that line comment-free (comment on its own line above).

**The load-bearing finding — resolution vs A/V sync on the one shared 16GB GPU.** `MUSETALK_SIZE` (output frame px,
couples server+client+`video_out`) was pushed 256→512→768→**1024** chasing sharpness. Isolated per-segment profiles
showed huge render headroom at every size (e.g. 1024/BASE_MAX768: ~129ms per 8-frame seg vs the 667ms/12fps budget),
**but under REAL live load — MuseTalk sharing the card with CosyVoice's vLLM — 1024 dropped to ~10fps effective**
(`[avatar timing] … 10.1 fps … LIPS BEHIND`), and in `steady` mode the voice is paced to the lagging video → audible
**voice lag**. Retesting 1024 at `fps=12` + `BASE_MAX=768` + `2M` bitrate still lagged live. **Verdict: `MUSETALK_SIZE=512`
is the practical ceiling for lag-free real-time on this box** (still 4× the pixels of the original 256; sharper studio,
no drift). Also fixed a genuine drift source found en route: the avatar server had been relaunched at `MUSETALK_FPS=14`
while `.env`/pipeline used `12` — **one fps everywhere is load-bearing** (server stride, client release clock,
`video_out_framerate`), now aligned at 12. Lesson (again): the isolated render profile passes what the live shared-GPU
run rejects — trust the live `[avatar timing]`/`hold` under contention, not the isolated per-segment cost. Structural
fix for higher res stays the dedicated avatar GPU.

**State.** All the above is **UNCOMMITTED** on `feat/ttfo-first-clause-split` (new `local_services/nimbus_client/`,
`main.py` `_install_nimbus_client` + `/client/say` + `/client/transcript` + `_TranscriptStore`/observer, `.env` avatar
ref + provider unpin + `MUSETALK_SIZE=512`/`BASE_MAX=768`, `run.ps1` propagates `MUSETALK_BASE_MAX`, new `assets/avatar_*`).
Open the custom UI at **`/nimbus/`** (trailing slash); `/client/` prebuilt is the untouched fallback.
_(Committed to `main` in the 12th session — see P37.)_

---

## P37 — `/nimbus/` chat: no USER bubbles (streaming-bubble work) + "a lot of bubbles" + mic mute + single-connection + no-interrupt ✅ SHIPPED (2026-07-08, 12th session)

**Symptom.** The `/nimbus/` chat showed the bot's reply bubbles but **never** the user's spoken turns. Then, once
user bubbles worked, a single spoken sentence produced **several** user bubbles. Plus two requested features: clicking
the mic should MUTE (not disconnect), and the bot should not be interruptible.

**Root cause — the missing user bubble.** The read-only transcript observer (P36) committed a user line with
`if getattr(frame, "finalized", True): store.add("user", frame.text)`. But `TranscriptionFrame.finalized`
**defaults to `False`**, and Deepgram's *streaming* STT only sets it `True` on an explicit `finalize()` — which the
active turn strategy never requests. Proven live with a temp probe: `[transcript-dbg] FINAL finalized=False '明天'`. So
the guard was False on every user frame → **every** user bubble was dropped (bot bubbles use the
`LLMFullResponseStart/End` bracket, unaffected). Subtle trap: pipecat's base **segmented** STT DOES flip
`finalized=True` in its `push_frame`, so the same code works for segmented providers (sherpa/funasr) but silently
fails for streaming Deepgram — a provider-dependent gap that a unit test with a hand-built frame (default `finalized`)
would also miss.

**Fix — commit on frame TYPE, not `finalized`.** Any `TranscriptionFrame` is a committed user line; the in-progress
text arrives as the separate `InterimTranscriptionFrame`. So: interims → `store.set_partial()` (the streaming live
bubble); `TranscriptionFrame` → accumulate the user turn.

**"A lot of bubbles" → one per turn.** Deepgram emits a `TranscriptionFrame` **per speech pause**, so committing one
bubble per frame produced a bubble per pause. Instrumented the per-turn frame order (`[seq-dbg]`):
`Interim(s) → TranscriptionFrame(s) → UserStoppedSpeaking → LLMFullResponseStart`. Fix: accumulate every finalized
segment into `_user` (which also feeds `set_partial` so the live bubble keeps growing across pauses), and commit ONE
`user` bubble at **`LLMFullResponseStartFrame`** — the bot beginning to reply is the provider-independent "user turn
complete" signal (chosen over `UserStoppedSpeakingFrame` because sherpa's `VADUserStoppedSpeakingFrame` is NOT a
subclass of it, so a stop-frame anchor would be fragile across STT providers). Segments join with **no separator** for
zh/th (a space reads as a mid-sentence break), a space for word languages. Caveat: a long utterance the turn analyzer
splits into two *turns* (two bot replies) is still two bubbles — correct.

**Streaming bubble transport.** `_TranscriptStore` gained a `_partial` slot (`set_partial`/`clear_partial`/`partial`
property, not seq'd); `/client/transcript` returns `"partial": {text, updatedAt} | null` beside `items`. Client polls
at **200ms** (was 900ms), renders one `.msg.user.live` bubble (blinking caret via `@keyframes caretPulse`, 82%
opacity), swaps it for the committed bubble when the `user` item arrives, and has a 2s client-side staleness guard for
a trailed-off partial. Verified live: interim `台 → 台北明天的天 → 台北明天的天氣怎麼樣` streamed, then one committed
`user '台北明天的天氣怎麼樣'`.

**Mic MUTE toggle.** `micBtn` click while connected toggles mute via `micStream.getAudioTracks()[t].enabled=false`
(a disabled track sends silence → STT hears nothing) with a red ring + diagonal `::after` slash + status "Muted" + aria
update; Disconnect stays its own button. Resets on connect/teardown; the bot-audio analyser's Listening/Speaking label
respects the mute flag.

**Single-connection policy.** pipecat's runner spawns a fresh `bot()` per `/api/offer`, so two browsers would fight
the single-client avatar server (`:8002`) on the one shared GPU. Now a module-global `_active_connection` is claimed at
the top of `bot()` and the previous connection is `await old.disconnect()`'d **before** the new pipeline is built (so
`:8002` is released first); `run_bot(transport, conn)` threads the conn; `_on_disconnected` clears the slot only if
it's still ours (a newer client already claimed it = us being kicked). Proven: probe#1 → `video frames: 0` (kicked),
probe#2 → `video frames: 420` clean; log `New WebRTC offer -- disconnecting the previous session`.

**No-interrupt mode (`ALLOW_INTERRUPTIONS`).** New `.env` knob (default `1` interruptible; live `.env` set to `0`).
`0` = the bot always finishes; user speech during playback never cancels it. The interruption is broadcast by the user
**turn-start** strategy when `enable_interruptions` is True (`user_turn_processor._on_user_turn_started` →
`broadcast_interruption()`), so the fix builds the default start strategies with it **off**:
`UserTurnStrategies(start=[VADUserTurnStartStrategy(enable_interruptions=False),
TranscriptionUserTurnStartStrategy(enable_interruptions=False)])`, keeping the default smart-turn STOP strategy. This
is deliberately NOT the `AlwaysUserMuteStrategy` mic mute (`ECHO_GUARD`), which is broken under steady sync (P11); the
strategy flag has no mute state machine so it's safe under steady. `main.py` now builds one shared
`LLMUserAggregatorParams` for echo-guard + no-interrupt. Proven live: user "started speaking" twice mid-reply → **zero**
`broadcasting interruption` (vs one per user-start in every prior session). Residual (unfixed): with a live mic + no
headphones, barge-in/echo speech is no longer a cut-off but can still be transcribed and answered AFTER the bot
finishes; the proper cure is a TTS-frame-based mic mute (mute on `TTSStarted`/unmute on `TTSStopped`, P11's future
option).

**State.** SHIPPED and **committed to `main`** (fast-forward from `feat/ttfo-first-clause-split`; main was a strict
ancestor). Files: `pipeline/main.py` (observer + `_partial` + single-connection + no-interrupt),
`pipeline/config.py` (`allow_interruptions`), `.env` (`ALLOW_INTERRUPTIONS=0`),
`local_services/nimbus_client/index.html` (live bubble + mute). All verified with `scripts._webrtc_probe`.
