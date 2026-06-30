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
