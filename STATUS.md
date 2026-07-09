# VisualLLm — Project Status & Next Steps

_Last updated: 2026-07-08 (**12th session — the `/nimbus/` client made real: streaming STT bubble, a "no user bubble"
bug root-caused + fixed, one-bubble-per-turn, a mic MUTE toggle, single-connection enforcement, and a no-interrupt
mode. Then THIS WHOLE BASELINE COMMITTED + fast-forwarded to `main`. See `docs/PROBLEMS-AND-FIXES.md` P37.**
(1) **Streaming STT bubble** — the user's speech now fills a LIVE bubble word-by-word as STT hears it (interim
results → a `_partial` slot on the transcript store → the client polls it at 200ms and renders one growing bubble
with a blinking caret), then swaps to the committed bubble. (2) **THE "no user bubble" BUG (P37)** — the transcript
observer gated user commits on `frame.finalized`, but Deepgram's STREAMING path leaves `finalized=False` unless an
explicit finalize was requested → **every** user bubble was silently dropped (only bot bubbles showed). Fixed by
committing on the frame TYPE (any `TranscriptionFrame` is a user line; interims arrive as the separate
`InterimTranscriptionFrame`). (3) **One bubble per turn** — Deepgram emits a `TranscriptionFrame` per speech PAUSE,
so committing per-frame made "a lot of bubbles"; now the segments accumulate and commit as ONE user bubble at
`LLMFullResponseStart` (the provider-independent "user turn done" signal), joined with no space for zh/th. (4) **Mic
MUTE toggle** — clicking the mic while connected now toggles mute (disables the outgoing audio track → red ring +
slash + "Muted"; Disconnect stays its own button), instead of tearing down. (5) **Single-connection policy** — a new
WebRTC offer DISCONNECTS the previous session first (`_active_connection` global, `old.disconnect()` before building
the new pipeline) so two clients never fight the single-client avatar server / shared GPU. Proven: probe #1 kicked to
0 frames, probe #2 clean 420 frames. (6) **No-interrupt mode** — `ALLOW_INTERRUPTIONS=0` (new `.env` knob, default 1)
makes the bot ALWAYS finish its reply; user speech during playback never cancels it. Done via the turn-START
strategies' `enable_interruptions=False` — NOT the P11-broken mic mute, so it's safe under steady. Proven: user
"started speaking" twice mid-reply, **zero** `broadcasting interruption`. All verified live with `scripts._webrtc_probe`.
**Baseline is now on `main` (fast-forward from `feat/ttfo-first-clause-split`).**)_
<!-- prior handoff -->
_11th-session detail: (**11th session — a custom "Nimbus AI" web client shipped (figma-to-code) + the shared-GPU
resolution ceiling nailed down. See `docs/PROBLEMS-AND-FIXES.md` P36.** (1) **New `/nimbus/` UI** — a self-contained
vanilla-JS page (`local_services/nimbus_client/`, full-screen weather-anchor avatar + glass chat), served ADDITIVE by
`_install_nimbus_client` (StaticFiles, no-store); speaks the same `POST /api/offer` signaling as the prebuilt bundle,
so no build step and `/client/` stays the untouched fallback. Live A/V verified headless (offer 200, both tracks,
frames decode); Connect/**Disconnect** buttons + loading overlay. (2) **Text send + transcript bubbles** — `POST
/client/say` injects a typed turn (`LLMMessagesAppendFrame`→`_active_task`); `GET /client/transcript?since=N` feeds
chat bubbles from a READ-ONLY `BaseObserver` on the task (bot `LLMTextFrame`s + user `TranscriptionFrame`s, no
pipeline structural change). (3) **Avatar = the weather anchor** (`AVATAR_REF=assets/avatar_studio_match.png`, a
top-anchored square-padded studio crop; MuseTalk forces square output so wide shots get padded, `object-fit:cover`
crops the pad). (4) **LLM 429 dead-end FIXED** — `OPENROUTER_PROVIDER_ONLY` UNPINNED (empty) so a free-Groq-tier 429
auto-fails-over to deepinfra/novita/google-vertex instead of killing the turn; gotcha — the `.env` loader does NOT
strip an inline `#` from a value. (5) **Resolution vs sync ceiling** — `MUSETALK_SIZE` pushed to 1024 for sharpness
but under REAL CosyVoice GPU contention render fell to ~10fps → steady-mode voice lag; reverted to **512** (the
lag-free ceiling on this one 16GB card, 4× the original 256px). Fixed a real drift bug found en route: server was at
`MUSETALK_FPS=14` while pipeline used `12` — one fps everywhere or drift, now aligned. Lesson (again): the isolated
render profile passes what the live shared-GPU run rejects.)_
<!-- prior handoff -->
_10th-session detail: (**10th session — last mile made honest (measure.py now loads `.env` → browser est 0.40→0.15;
corrected local turn 1.86→0.80→0.15→2.81s) + a one-click "Measure turn" button (`MEASURE_BUTTON=1`) that fires a real
turn and times click→ear in-page; found real BROWSER turns log `[TTFO] count:0` (transcription turn-stop emits no
user-stop frame → production TTFO blind to real users); cross-device clock handled (server-clock beacon anchor). See
the top HANDOFF.** Prior **9th session — measure system now reports TRUE latency-to-the-user's-ear via a
per-stage waterfall; see that HANDOFF.** Prior **8th session — local OFFLINE STT shipped + two live root-causes found.**
(1) **Local sherpa STT integrated** onto `feat/ttfo-first-clause-split` (surgical port from `feat/offline-stt-sensevoice`,
commits `5bacfa8`+`df278fd`; Deepgram still default, `STT_PROVIDER=sherpa` opt-in, CPU/~0 VRAM). Fixed both open
items: `[TTFO]` count:0 (meter armed only on `UserStoppedSpeakingFrame`, sherpa drives turns with
`VADUserStoppedSpeakingFrame`), and the real blocker — sherpa's bot-speech pause STRANDED under steady (BotStopped
never fires, P11 mechanism) so the mic died after the greeting; gated the pause behind `echo_guard` (default off).
Probe-verified: sherpa zh turn → LLM → avatar, `[TTFO] count:1 ~2.1s pass`. `docs/PROBLEMS-AND-FIXES.md` **P29**.
(2) **CUDA graphs (P27) REVERTED to eager** (`COSYVOICE_VLLM_EAGER=1`): live they made per-turn latency **very
inconsistent** (shape-specific graphs + variable reply lengths → capture/JIT spikes; rigid replay fights MuseTalk
on the shared GPU). Eager = ~2.0s first-chunk baseline but STEADY; for a live avatar consistency > fast-avg. **P31**.
(3) **zh "avatar-first / voice-delayed" root-caused to the FILLER opener** (P26) colliding with zh's short-piece
splitting — NOT raw TTS speed (zh TTFB ≈ en ~0.9s). `FILLER_WORDS=0` (or filler en-only) fixes it. **P30**.
PROCESS NOTE: don't hand-restart the pipeline while the launcher runs — it creates a two-manager fight for the
single-client avatar server / shared GPU (muddied this session's A/B). Use the launcher or the config panel
`:7870` /restart as the single manager. Prior 7th session — TTS first-chunk cut ~2.0→~0.85s live, the compute lever STATUS had
flagged as "the wall"**: vLLM **CUDA graphs** (`COSYVOICE_VLLM_EAGER=0` default, dominant) + a tighter streaming
poll (`model.py` 0.1→0.02s) — isolated TTS TTFB −30–40% (en 1.05→0.75/2.03→1.26; zh 1.55→1.05/2.36→1.42), live
end-to-end TTFO all PASS <3s (en ~2.0, zh ~2.25s). Survives live where `hop` didn't — it makes the same-size
first chunk arrive sooner, not smaller (no lead-cushion erosion). Costs: +30–60s boot (graph capture), occasional
one-time JIT spikes; VRAM unchanged. Cosyvoice-repo changes uncommitted per pattern. `docs/PROBLEMS-AND-FIXES.md`
P27. **Then Lever 1 (flow-matching TRT) tested & REJECTED** — fp32 engine builds/runs on TRT 11 but zero TTFB win
(flow isn't the first-chunk bottleneck; vLLM prefill+decode dominates, already CUDA-graphed); env-gated OFF. TTS
first-chunk now near its floor for this stack. Prior 6th session — **filler-word "thinking" opener SHIPPED as the new baseline**
(`FILLER_WORDS=1`: the turn opens on a rotated "嗯，讓我想一下喔，…" so the avatar starts talking ~0.7s sooner —
a PERCEPTION win, since TTFO counts time-to-first-sound and that sound is the filler; the real answer arrives
slightly later); **both P20 turn-start levers (`MUSETALK_HP_STREAM` GPU stream priority, `COSYVOICE_PACE_RATE`
stagger) TESTED & REJECTED** — the TRT render isn't GPU-starved at turn start, both chased a non-bottleneck
(P25/P26). Prior 2026-07-04 (5th session) — **zh TTFO ~4.1→~3.1s**: `COSYVOICE_FIRST_HOP_ZH` reverted 5→0 (live
A/B, resolves P19's caveat) + the **zh comma-only first-piece split** shipped (`COSYVOICE_FIRST_PIECE_ZH=1`,
long-opener turns 4.8→3.1–3.4s, P23); `MUSETALK_LEAD_FRAMES=8` measured as the first all-under-3s candidate
(zh 3.03/en 2.48 median) but **REJECTED by the user's live eye later that day** — every lead below 14 shows
delay or avatar freezes; lead=14 is final (P22 addendum); + **phone-loudspeaker client fix**
(`CLIENT_FORCE_SPEAKER`, P24). `LANGUAGE=zh` live. Prior (4th session) — **LLM cloud hop pinned to Groq** (`OPENROUTER_PROVIDER_ONLY=Groq`);
**model baseline = `meta-llama/llama-4-scout`** (same speed as llama-3.3-70b, better/cleaner Traditional zh,
~5× cheaper — $0.11/$0.34 vs the 70b's real Groq price $0.59/$0.79). LLM hop halved + tail killed (zh 1.64→0.80s,
en 1.07→0.67s) but end-to-end TTFO only modestly down (TTS + steady-hold now dominate); + zh short-first-sentence
prompt (~0.3–0.5s); + **goal tightened to <3s** (en ~3.5s / zh ~4.3s now sit over the bar). `live` sync REJECTED
by the user (voice leads lips ~1–2s). All uncommitted. See the 4th-session block below + P21. Prior (3rd session)
— **FP8 quantization = DEAD END, proven**: tried to shrink the
MuseTalk render (fix the "avatar moves before voice" turn-start lag) via FP8; quality perfect (SSIM 0.99997)
but 4.5x SLOWER — TRT 10.13 has no FP8-conv kernels for Blackwell sm_120 (NVRTC compile failures), ruled out
the per-channel-scale hypothesis with a per-tensor tiebreaker. No config shipped, env cleanly rolled back.
Live measurement re-confirmed the shared-GPU wall (video ~2.7s behind the voice at turn start on a long reply,
even WITH TRT — the offline "flat drift" doesn't survive real CosyVoice contention). Handoff = two untried
non-hardware levers, **Lever 1 (GPU stream priority)** + **Lever 3 (stagger the bursts)** — see the 3rd-session
block below + `docs/PROBLEMS-AND-FIXES.md` P20. Prior (2nd) session: TTFO hop×lead sweep, negative, baseline
stands (P19). (2026-07-02): Chinese TTS fixed (RAS + "pro" voice); avatar baseline `MUSETALK_TRT=1`, LLM cloud
gemini-2.5-flash-lite; TTFO ~4.6s→~3.2s via the first-clause split `COSYVOICE_FIRST_PIECE=1`.)_

## ⭐ HANDOFF → next session (2026-07-08, 12th): `/nimbus/` client made real (streaming bubble + no-bubble bugfix + one-bubble-per-turn + mic mute + single-connection + no-interrupt); BASELINE COMMITTED TO `main`

**Context:** hardened the custom `/nimbus/` client into a usable UI and locked the whole baseline onto `main`.
Everything below was verified live with `python -m scripts._webrtc_probe --mic output/_zh_q_wx.wav --lead 8` (drives a
real turn through STT→LLM→TTS→avatar) + a fast HTTP poller of `/client/transcript`.

**What shipped (all in `pipeline/main.py` + `local_services/nimbus_client/index.html`, additive; `pipeline/config.py` + `.env` for the one knob):**

1. **Streaming STT bubble.** The user's in-progress speech fills a LIVE bubble word-by-word. Server: the transcript
   store gained a `_partial` slot (`set_partial`/`clear_partial`/`partial`), set from `InterimTranscriptionFrame`s;
   `/client/transcript` returns `"partial": {text, updatedAt} | null` beside `items`. Client: one `.msg.user.live`
   bubble (blinking caret, 82% opacity) polled at **200ms** (was 900ms), swapped for the committed bubble on commit;
   2s client-side staleness guard for a trailed-off partial.

2. **THE "no user bubble" bug — root cause + fix (P37).** User bubbles NEVER showed (only bot). The observer gated
   the user commit on `if getattr(frame, "finalized", True)`, but `TranscriptionFrame.finalized` **defaults False** and
   Deepgram's streaming path only sets it True on an explicit finalize() (which this turn strategy never requests) —
   proven live: `[transcript-dbg] FINAL finalized=False '明天'`. So every user line was dropped. **Fix:** commit on the
   frame TYPE — any `TranscriptionFrame` is a committed user line; interims are the separate
   `InterimTranscriptionFrame`. (pipecat's base STT DOES flip `finalized=True` for SEGMENTED providers, so the old code
   worked for those but not streaming Deepgram — a silent provider-dependent gap.)

3. **One bubble per turn.** After (2), the user reported "a lot of bubbles": Deepgram emits a `TranscriptionFrame`
   per speech PAUSE, and we committed one bubble each. Instrumented the frame order (`[seq-dbg]`):
   `Interim(s) → TranscriptionFrame(s) → UserStoppedSpeaking → LLMFullResponseStart`. **Fix:** accumulate all finalized
   segments into `_user` (also feeding the live partial), and commit ONE `user` bubble at `LLMFullResponseStartFrame`
   (the bot starting to reply = the user's turn is unambiguously done; provider-independent — `VADUserStopped` is NOT
   a subclass of `UserStopped`, so anchoring on the stop frame would be sherpa-vs-Deepgram fragile). Segments join with
   no separator for zh/th, a space otherwise. NOTE: a genuinely long utterance the turn-analyzer splits into two
   *turns* (two bot replies) will still be two bubbles — correct, that's two turns.

4. **Mic MUTE toggle.** `micBtn` click while connected now toggles mute via `track.enabled=false` on the outgoing
   audio (red ring + diagonal slash, status "Muted", aria updated), NOT teardown — Disconnect is its own button. Resets
   on connect/teardown. The bot-audio analyser's Listening/Speaking label respects the mute flag.

5. **Single-connection policy.** A new `/api/offer` spawns a fresh `bot()` (pipecat runner does one per offer), which
   would let two browsers fight the single-client avatar server. Now a module-global `_active_connection` is claimed at
   the top of `bot()` and the PREVIOUS connection is `await old.disconnect()`'d **before** the new pipeline is built
   (so `:8002` is released first). `run_bot(transport, conn)` takes the conn; `_on_disconnected` only clears the slot if
   it's still ours (a newer client may have already claimed it = us being kicked). Proven: probe#1 → `video frames: 0`,
   probe#2 → `video frames: 420` clean; log shows `New WebRTC offer -- disconnecting the previous session`.

6. **No-interrupt mode (`ALLOW_INTERRUPTIONS`).** New `.env` knob (default `1` = interruptible; set to **`0`** in the
   live `.env` per the user's ask). `0` makes the bot ALWAYS finish its reply — user speech during playback never cuts
   it off. Implemented via the turn-START strategies' `enable_interruptions=False`
   (`UserTurnStrategies(start=[VADUserTurnStartStrategy(enable_interruptions=False),
   TranscriptionUserTurnStartStrategy(enable_interruptions=False)])`, default smart-turn STOP strategy preserved). This
   is the flag that broadcasts the barge-in — **NOT** the `AlwaysUserMuteStrategy` mic mute, which is BROKEN under steady
   sync (P11); this has no mute state machine so it's safe under steady. `main.py` now builds one
   `LLMUserAggregatorParams` shared by echo-guard + no-interrupt. Proven: user "started speaking" twice mid-reply →
   **zero** `broadcasting interruption` (every prior session logged one per user-start).

**Git:** the whole baseline (20 branch commits + all this session's + prior uncommitted Nimbus/measure work) was
committed on `feat/ttfo-first-clause-split` and **fast-forwarded to `main`** (main was a strict ancestor — clean ff,
no divergence). Repo is PUBLIC (`Triple3Pww/visualllm`); `.env` is gitignored (no secrets).

**Open / next:**
- With the mic always live and no headphones, no-interrupt stops the CUT-OFF but barge-in/echo speech can still be
  transcribed and answered AFTER the bot finishes (bot may reply to its own echo). The proper fix is a TTS-frame-based
  mic mute (mute on `TTSStarted` / unmute on `TTSStopped`, P11's noted future option) — untried.
- Real-mic still the user's to validate (use headphones); probes drive a synthetic wav.

## ⭐ HANDOFF → next session (2026-07-06, 10th): last-mile made honest + a one-click "Measure turn" button; real browser turns are blind to `[TTFO]`

**Context:** picked up the 9th-session waterfall and drove the last two columns (the last mile) to real numbers.
Everything below is in `scripts/measure.py`, `pipeline/main.py`, `.env` — **`pipeline/metrics.py` still UNTOUCHED**.

**1. The "1.26s last mile" was ~40% a MEASUREMENT ARTIFACT.**
- `measure.py` never loaded `.env`, so `os.getenv("CLIENT_JITTER_BUFFER_MS","400")` used the hardcoded **400**, not the
  real **150** → the browser ESTIMATE row was overstated ~250ms. Fix = `load_dotenv(ROOT/".env")`. Corrected clean
  headless zh turn: **TTFO 1.86s → transport +0.80s (probe) → browser +0.15s (est) → ear 2.81s** (was 2.22→0.86→0.40).
- The `transport +0.80s` is a same-box LOOPBACK number — mostly NOT network; it's WebRTC/Opus encode + the onset
  detector waiting out CosyVoice's sub-threshold **zh leading breath** (P34). So the real improvable last mile for a
  LOCAL viewer is small (~0.15–0.8s).

**2. Real BROWSER turns log `[TTFO] count:0` — the production metric is BLIND to real users.** Confirmed root cause:
pipecat's transcription turn-stop path (`llm_response_universal`, logged **"strategy: None"**) broadcasts NO
user-stop frame the `TtfoMeter` arms on, and a noisy real mic yields no Silero VAD stop either (only a clean-silence
headless WAV fires `TurnAnalyzerUserTurnStopStrategy` → `VADUserStoppedSpeakingFrame`). `TranscriptionFrame` is
consumed by the aggregator before the meter's post-avatar position. A meter-side fix needs fragile version-specific
pipecat turn surgery → **fixed in `measure.py` instead:** new `parse_browser_turn()` anchors `--from-browser` on the
`[client-playout]` beacon (not a stale `[TTFO]`) and derives the turn's TTFO itself.

**3. THE BUTTON (`MEASURE_BUTTON=1`, default OFF) — the robust way to measure a real browser turn.** A "Measure turn"
button injected into `/client/`. On click (a USER GESTURE → `ctx.resume()`, the reason it works where the passive
`CLIENT_PLAYOUT_PROBE` never fired live) it taps the played bot audio, fires ONE real turn via **`POST
/client/measure-turn`** (server injects `LLMMessagesAppendFrame(run_llm=True)` with a fixed question into the live
`_active_task` → full LLM→TTS→avatar turn), times **click→first-voice-onset** and shows it in-page ("heard N ms"),
and beacons the onset to `/client/playout`. **Verified live** — the user clicked it 3× (each: `[measure-turn]
injected turn` + `Bot started speaking` + a real `"src":"button"` beacon). A JS brace bug (missing `}` → `Unexpected
token 'catch'` → whole script died → "no button") was caught with Playwright and fixed.

**4. CROSS-DEVICE CLOCK (important).** The user views `/client/` from a **separate device over Tailscale** whose clock
was ~14 min off the server — this breaks the whole same-box `Date.now()==time.time()` assumption (beacon−bot_started =
−834s garbage). Fix: `parse_browser_turn` anchors the last mile on the beacon's **server-clock log-arrival time**, not
the browser `Date.now()` in the body → offset-immune. **CAVEAT (honest):** for a REMOTE device that server-clock Δ =
last-mile-out + the beacon's network trip BACK, so it OVER-counts and SWINGS with the link (measured 0.94s then 1.99s
on Tailscale — real variance, not a bug). **The trustworthy figure is the button's IN-PAGE number** (click→ear, a
single browser-clock delta — no offset, no return-trip). Remote last mile is network/jitter-buffer dominated (lever =
`CLIENT_JITTER_BUFFER_MS` + the link), NOT the pipeline.

**Files (all UNCOMMITTED):** `pipeline/main.py` (`_install_measure_button`, `/client/measure-turn`, `_active_task`
global set in `run_bot`/cleared in `_on_disconnected`), `scripts/measure.py` (`load_dotenv`, `parse_browser_turn` +
`_build_turn` refactor, server-clock beacon anchor + freshness guard), `.env` (`MEASURE_BUTTON=1`, `CLIENT_PLAYOUT_PROBE=1`).

**OPEN:** (a) optionally beacon the button's click→ear DURATION (clock-immune) so `--from-browser` reports the accurate
experienced latency instead of the network-inflated server-clock Δ — needs a restart + reload + click; (b) whether to
arm the production `TtfoMeter` on the transcription turn-stop path so real-user turns log `[TTFO]` (deliberate pipecat
work); (c) commit decision for this session's changes.

---

## ⭐ HANDOFF → next session (2026-07-06, 9th): measure system now reports the TRUE per-stage latency all the way to the user's ear

**What shipped.** The measure harness (`scripts/measure.py`) now decomposes a turn into a **per-stage latency
waterfall that sums to a true end-to-end number** — closing the gap where `[TTFO]` stopped at
`BotStartedSpeakingFrame` (pipeline starts pushing audio) and the last mile (transport pacing, steady lead-hold,
WebRTC encode, network, jitter buffer, playout) was invisible. **Core mechanism:** the headless probe and the
pipeline run on the SAME box = the SAME wall clock, so `t0.timestamp()` (log t0) vs the probe's `time.time()` vs
the browser's `Date.now()` are one epoch — stitch them and every client arrival becomes a real offset from t0.
Spec: `docs/superpowers/specs/2026-07-06-measure-end-to-end-latency-design.md`; plan:
`docs/superpowers/plans/2026-07-06-measure-end-to-end-latency.md`.

**THE TABLE — a clean measured zh turn (2026-07-06 01:52, full stack up):**

| Stage (from t0 = user stopped) | Δ | cum | source |
|---|---|---|---|
| STT finalize -> LLM | +0.00s | 0.00s | assumed (pre-warmed) |
| LLM first token | — | — | **unknown** (synthetic-mic VAD split; real turn would populate) |
| LLM -> TTS (1st-clause flush) | +1.04s | 1.04s | log |
| TTS synth first chunk | +0.68s | 1.72s | log |
| TTS -> bot-start (steady lead-hold) | +0.50s | **2.22s = TTFO** | log |
| **Transport + encode + network** | **+0.86s** | 3.08s | probe *(was invisible)* |
| **Browser jitter + decode + playout** | **+0.40s** | **3.48s** | est (400ms jitter buffer) |
| **END-TO-END, user hears** | | **3.48s** | |

**Headline: `[TTFO]`=2.22s but the user HEARS the voice at 3.48s — a +1.26s last mile that was previously
unmeasured.** (Transport +0.86s is MEASURED at a headless client; browser +0.40s is ESTIMATED from
`CLIENT_JITTER_BUFFER_MS` until a real browser beacon fills it.)

**Two last-mile sources (dual, like the existing `lip_offset` offline/webrtc):**
1. **Headless (always-on):** `python -m scripts.measure --mic <wav>` drives a turn, and the audio pump now records
   `(epoch, rms)` per frame; `answer_onset_epoch()` finds the first sustained energetic frame after t0 = the answer
   reaching the client. Gives the `probe` transport row automatically, no browser.
2. **Real browser (opt-in, `CLIENT_PLAYOUT_PROBE=1`, default OFF):** a `<head>`-injected AnalyserNode taps the bot
   audio track and beacons `[client-playout] {"ev":"audio-onset","t":<ms>}` to a new `/client/playout` endpoint the
   first time RMS crosses threshold (re-arms per turn). `python -m scripts.measure --from-browser` parses that beacon
   into the `browser` row (labeled `browser+net` when there's no headless probe, since that Δ then absorbs transport).

**Verification state (what I proved vs what's left):**
- **Headless path: FULLY verified live by me** (the table above is a real run). It also EXPOSED + fixed a real bug
  the offline tests couldn't: a synthetic-mic turn matched an LLM-TTFB line from before t0 → rendered a negative
  `-1.34s` latency; now any negative (pre-t0) anchor renders **unknown** (a stage can't finish before the turn
  starts). 7/7 pure tests green.
- **Browser beacon path: server side proven end-to-end via a synthetic beacon** — armed `CLIENT_PLAYOUT_PROBE=1`,
  restarted via the config panel, confirmed (a) the probe script IS served on `/client/`, (b) `/client/playout`
  returns 204 + logs, (c) `--from-browser` parses it into the waterfall (`browser+net +0.36s cum 2.50s`). **The ONLY
  unexercised link is a real browser's AudioContext firing on real voice** — needs a human at the page with a mic.
  When you do it: `CLIENT_PLAYOUT_PROBE=1` in `.env`, restart, open `/client/`, one turn, `--from-browser`.

**GOTCHAS for next session:**
- **Synthetic-mic drives VAD-split** (the WAV's internal pause) → the turn's question text merges two sub-turns
  (`明天'}...台北攀不開下雨`) and the LLM-TTFB anchor can land pre-t0 → the **LLM row shows `unknown`**. That's the
  guard working, NOT a bug; a real human turn populates it.
- **Live browser TTFO does NOT log for rapid/overlapping (barge-in) turns** — `pipeline/metrics.py`'s TtfoMeter only
  fires on a clean user-stop->bot-start pair (saw bot stop->start 2ms apart on fast turns → no `[TTFO]`). Pre-existing
  behavior; `metrics.py` was deliberately left UNTOUCHED by this feature.
- **`measure.py` reads the LAST `[TTFO]` in the log**; the new **staleness guard** blanks `client_arrival` (with a
  warning) if that turn is older than `duration+tail+15s`, so a stale t0 can never silently produce a wrong number.

**Commits + branch state:** 10 commits on `feat/ttfo-first-clause-split` (`e30aa77` spec, `4244a6e` plan, then
`481c8be`..`e6f5bba` code), reviewed per-task + a final opus whole-feature review (**Ready-to-merge: YES, no
Critical/Important**). **NOT on `main`** — the user chose keep-on-branch (the browser-beacon commit `4f1d883`
hard-depends on held infra `f3ebb3f`'s client-patch middleware, so "measure only" isn't cleanly separable; it all
lands together when the branch is validated live + merged). Stack left healthy, probe OFF, `.env` byte-clean.

**OPEN:**
- **Real-browser onset test** (the one human-only verification link above).
- **`docs/PRESENTATION.md`:** my spec commit `e30aa77` swept up a pre-staged 259-line deletion of it (it was already
  `D` in the index at session start). Decide keep vs. split back out — user was asked, deferred.

---

## ⭐ HANDOFF → next session (2026-07-05, 8th): zh turn-start "breathing sound" diagnosed = CosyVoice's leading breath; trim REJECTED, no-trim is BASELINE

**What it was.** The soft breath + pre-answer mouth-move on zh turns is **CosyVoice's zero-shot leading breath** —
25–610 ms of −34..−68 dB pre-speech content baked into ~every zh piece's waveform (measured by probing `:8001`
directly). The avatar lip-syncs off a Whisper of the waveform (like P33) so the mouth moves over it. NOT the filler
(off), NOT the reference clip, NOT the LLM. Full diagnosis + evidence: `docs/PROBLEMS-AND-FIXES.md` **P34**.

**What we tried and dropped.** A start-of-turn leading-breath trim in `cosyvoice_tts.py` (arm on
`LLMFullResponseStartFrame`, stream-drop the lead below −33 dB). Offline-verified + zh TTFO fine (**2.72 s median**,
no regression), **but it broke live**: `np.frombuffer` on aiohttp's odd-sized chunks crashed the first piece →
"cosy only speaks a sentence each." **Reverted fully** (code back to HEAD, `.env` knobs removed). User's call:
*without trim is better* → the breath is accepted, **no-trim is the baseline**. If re-attempted, trim **server-side
in CosyVoice** (whole buffers), not in the byte-stream client.

**⚠️ Live note:** if the pipeline wasn't restarted after the revert, it may still hold the buggy trim in memory
(the single-sentence bug) — restart it (config-panel **Restart**) to load the clean baseline.

---

## ⭐ HANDOFF → next session (2026-07-05, later 8th): CUDA graphs CLOSED = keep eager (zh-lipsync cost); + config-panel graphs toggle + freeze logging

**TL;DR — three threads this session, all left in a safe, verified state; NOTHING committed (both repos).**

### 1. CUDA graphs — FINAL VERDICT: KEEP EAGER (`COSYVOICE_VLLM_EAGER=1`). Do NOT re-open as "no drawback."
The user asked "can we use graphs without the drawback?" I re-tested and first concluded "yes, no drawback" —
**that was measuring the WRONG side.** A TTS-TTFB variance probe (`cosyvoice-local-tts/_ttfb_variance.py`) shows
graphs are genuinely faster + lower-variance on the TTS stopwatch (96 samp under real MuseTalk render: graphs
1.29/2.23/0.37s vs eager 1.94/3.43/0.64s) — and the two memory-stated drawbacks (novel-shape capture spikes;
graphs-worse-under-contention) did NOT reproduce. **BUT the user's eye caught the real cost: zh LIPSYNC stops
matching the words with graphs ON.** Root cause, MEASURED (`cosyvoice-local-tts/_zh_audio_ab.py`, same zh
sentence ×5 each): graphs ON alter the zh AUDIO — duration median 8.92 vs 8.28s, longest internal silence 0.76 vs
0.68s, silence-frac 0.30-0.36 vs 0.22-0.33, more run-variance. Mechanism: the graph decode perturbs the
zh-critical **RAS** sampling (the P18 fix that stops zh looping on the silence token); MuseTalk lip-syncs off a
**Whisper of the waveform**, so a degraded zh waveform → mouth shapes that don't track the phonemes. en is spared
(no RAS reliance); render fps stayed ~14 throughout (NOT a render-starve). **This reconciles with P31's original
revert — the eye was right; the TTS-side probe structurally can't see the zh-audio cost.** `docs/PROBLEMS-AND-FIXES.md`
**P32** (the diagnose-then-fix TTS investigation) + **P33** (the zh-lipsync verdict). Spec:
`docs/superpowers/specs/2026-07-05-cuda-graphs-without-drawback-design.md`. Recurring lesson (now 3×): the probe
passes what the eye rejects.

### 2. Config panel (`:7870`/`:8444`) — refreshed + a CUDA-graphs toggle. VERIFIED end-to-end.
`server.py`+`index.html`: added the levers we actually tune (`STT_PROVIDER`, `OPENROUTER_PROVIDER_ONLY`,
`FILLER_WORDS`, `COSYVOICE_FIRST_PIECE`, +advanced) and fixed stale option lists (llama-4-scout, fps=14). NEW
**CUDA graphs toggle**: it rewrites `run_vllm_server.sh`'s `EAGER` default AND relaunches the WSL vLLM server
(the pipeline Restart can't reach WSL) via the launcher's detached-`wsl.exe` pattern. Round-trip tested (off→eager→
on→graphs, health-verified). Carries the P15 load-order/OOM caveat in its failure message.

### 3. Freeze-capture logging — NEW, both sides of the path. VERIFIED (browser beacon round-trips; watchdog not yet fired in anger).
The old logging sampled `hold`/`offset` ~1×/s server-side only, so a real freeze fell between samples / lived in
the browser. Added: (a) pipeline-side **`[avatar FREEZE]`** watchdog in `musetalk_video.py` (`MUSETALK_STALL_LOG_S`
default 0.4s; classifies render-starved vs delivery-side); (b) browser-side **`[video-stall]`** monitor in
`main.py` (`requestVideoFrameCallback` → beacons the *displayed* freeze to pipeline.log; `CLIENT_VIDEO_STALL_MS`
default 350, `CLIENT_VIDEO_STALL_MONITOR=0` to disable). To use after a freeze: `grep -E "FREEZE|video-stall" logs/pipeline.log`.

### OPEN / TODO for next session
- **Commit hygiene:** everything is UNCOMMITTED in BOTH repos (per the hold-for-live pattern). visualllm: `M`
  config_panel/{server.py,index.html}, musetalk_video.py, pipeline/main.py, + docs/memory + this STATUS; `??`
  the spec. cosyvoice-local-tts: `M` run_vllm_server.sh (comment only — value already eager), `?? _ttfb_variance.py`,
  `_zh_audio_ab.py`. Decide what to keep vs. the throwaway probes.
- **Latent bug (separate from the zh-graphs issue):** vendored `musetalk_server/vendor/MuseTalk/.../audio_processor.py:90`
  calls bare `exit()` when a segment's last whisper frame runs short → the segment is caught+dropped (returns []) =
  a sub-second freeze. Fix = pad the short clip instead of `exit()`. Did NOT touch it this session.
- The 18:47 freeze the user reported was NEVER root-caused (server logs were clean); the new logging should catch
  it on recurrence. Reproduce a zh turn on eager and watch for `[video-stall]`.

### STATE at handoff: graphs OFF/eager (script default + running WSL server both `EAGER=1`); full stack UP
(`:8001` eager vLLM, `:8002` MuseTalk TRT, `:7860` pipeline, `:7870` panel). NOTE the running MuseTalk/pipeline
were started by me/the launcher mid-session — a clean relaunch via `Run VisualLLm.exe` is the safe reset.

---

## ⭐ HANDOFF → next session (2026-07-05): BOTH P20 levers tested & REJECTED; the turn-start render is NOT GPU-starved

**Implemented + live-A/B'd both remaining P20 levers. Both fail; the P20 premise is wrong on the TRT baseline.**
Full data + mechanism: `docs/PROBLEMS-AND-FIXES.md` P25. Knobs kept in-tree, default-OFF (inert), marked dead.
- **Lever 1 — high-priority CUDA render stream (`MUSETALK_HP_STREAM`):** CATASTROPHIC live (TTFO ~12s, render
  collapses to ~1–2fps). Isolated offline it's byte-identical to baseline (~101ms/seg) — so it self-destructs
  ONLY when contending with CosyVoice: CUDA stream priority is intra-context, but MuseTalk (Windows) vs
  CosyVoice (WSL VM) are two processes under WDDM, which **thrashes** the shared GPU instead of honoring
  priority (cross-process priority needs CUDA MPS = Linux-only). Dead on Windows+WSL.
- **Lever 3a — CosyVoice turn-start throttle (`COSYVOICE_PACE_RATE`; the docs' knob never existed, built it):**
  useless + unstable. Didn't reduce the hold (already steady ~0.45s) and made it swing 0.39→5.33s (starves the
  avatar of audio to render). No upside.
- **THE FINDING (corrects P20):** on the TRT+GPU_COMPOSITE baseline the render is NOT starved at turn start
  (offline ~101ms/8-frame-seg ≈78fps; live hold is a rock-steady ~0.45s = the **structural `lead=14` fill**, not
  contention). Both levers chased a non-bottleneck. The dominant, variable TTFO cost is the **TTS first-chunk
  (CosyVoice prefill ~1.3–2.1s, scales with first-sentence length)**. Corollary: a **dedicated avatar GPU won't
  cut turn-start TTFO** (the hold is lead-fill, not contention) — it only helps long-reply drift (P16).
- **So the ONLY remaining TTFO lever is shrinking the TTS first-chunk** (extend the first-piece splits / shorter
  opener) or `lead` (eye-rejected <14). Fresh baseline this session was mostly already <3s (2.1–3.1s zh);
  the over-3s tail is entirely the TTS first-chunk on long openers.
  **→ DONE (7th session, P27):** shrank the TTS first-chunk by *compute*, not input — vLLM CUDA graphs
  (`COSYVOICE_VLLM_EAGER=0`) + tighter poll cut it ~2.0→~0.85s live (isolated TTFB −30–40%), TTFO now en ~2.0 /
  zh ~2.25s all-PASS.
  **→ Lever 1 (flow-matching TRT) TESTED & REJECTED (7th session):** installed TensorRT in WSL (cu13/**TRT 11**),
  ported CosyVoice's TRT-8/9 build helper to TRT 11 (3 patches), fp32 engine builds + runs clean — but **zero TTFB
  win AND ~26-40% SLOWER throughput** (controlled 10-run RTF A/B: RTF 0.31→0.40/0.43; `forward_estimator` hard-syncs
  the GPU ~20×/chunk, swamping fp32 fusion). Finding: the flow is NOT the first-chunk bottleneck (vLLM
  prefill+decode dominates, already CUDA-graphed), and TRT's sync overhead makes it a net loss here. fp16 (~1.9× on
  flow compute) would have to overcome that sync overhead too + needs an fp16-ONNX conversion + accuracy risk → not
  worth it. Env-gated `COSYVOICE_FLOW_TRT` default-OFF, inert. First-chunk is now genuinely near its floor for this stack; see the memory note. P27 + memory.

**Then — filler-word opener SHIPPED as the NEW BASELINE (`FILLER_WORDS=1`, `.env`).** The turn now opens on a
natural rotated "thinking" phrase ("嗯，讓我想一下喔，…") synthesized through the normal TTS path (one continuous
turn, zero screech risk — audio gaps ~60ms), so the avatar starts talking + lip-moving ~0.7s sooner (measured zh
**def 2.91→2.23s, wx 2.38→2.03s**; holds steady ~0.49s). **HONEST framing (the user asked how TTFO counts this):
TTFO measures time-to-first-SOUND, and that sound is now the filler — so the metric improves but the real ANSWER
arrives slightly LATER (queued behind the filler). It's a PERCEPTION/responsiveness win, not a real speedup.**
Tuning learned: fillers must be ~1.2s each so the first piece fills the `lead=14` cushion (a 0.3s "嗯，" opener
ballooned the hold 0.5→1.7s — the wx regression, fixed by the longer pool); `FILLER_WORDS_COUNT` chains more.
`first_piece_aggregator.py` (needs `COSYVOICE_FIRST_PIECE=1`), default-off code, `.env`=1. `docs/PROBLEMS-AND-FIXES.md`
P26. Stack left clean on this new baseline (cosy/avatar/pipeline up, filler on). Open follow-ups the user may
want: fire the filler only on slow turns (not trivial replies), a "first real answer" measurement alongside TTFO,
or the instant cached-audio version (~1s start, but the riskier raw-audio path).

## ⭐ HANDOFF (2026-07-04 evening) [SUPERSEDED by 2026-07-05 above — levers now tested]: the P20 turn-start levers

**State on exit:** all config levers are settled and live — hop=0 both langs (P22), en first-clause split +
zh comma-only split (P23), Groq pin + `llama-4-scout` (P21), `MUSETALK_SYNC_MODE=steady`,
**`MUSETALK_LEAD_FRAMES=14` FINAL (CLOSED — the user live-eye tested every value below 14: delay or avatar
freezes; never re-try)**, and `MUSETALK_FPS=14` (user's own change, propagated consistently). Fresh 4-turn zh
baseline at lead=14: **median 3.36s (2.91/3.05/3.67/3.77) vs the 3s target** — and in every turn the budget is
LLM ~0.7–0.9s | **CosyVoice first-chunk 1.44–2.00s (dominant)** | avatar hold 0.76–1.04s. A lone lead=8 probe
turn hit 2.50s (hold →0.48s), proving the hold cost is real — but that lever is closed; the remaining path to
<3s attacks the TTS first-chunk + the turn-start GPU collision (P20).

**Do next, in order (full impl detail in P20's HANDOFF section):**
1. **Lever 3a — turn-start stagger (preferred, low risk):** throttle CosyVoice's generation nearer real-time
   at turn start so its opening vocoder burst stops starving MuseTalk's first render segment. Knob:
   `COSYVOICE_PACE_RATE` (1.3) lives in the **cosyvoice repo** (`E:\Claude\cosyvoice-local-tts`), likely needs
   a dedicated turn-start-only throttle rather than a global rate cut. Variant 3b (delay the first render feed
   ~150–250ms in `musetalk_video.py`) trades lip-start delay — try 3a first.
2. **Lever 1 — CUDA stream priority (~10-line probe, may be dead on arrival):** high-priority
   `torch.cuda.Stream` around `render_segment` in the MuseTalk server (exact snippet in P20). Honest risk:
   WDDM does not strongly honor CROSS-PROCESS priority (CosyVoice=WSL, MuseTalk=Windows; MPS is Linux-only).
   If live `hold=` doesn't move, declare it dead and move on.
3. **Structural: dedicated avatar GPU** — the only guaranteed cure; avatar working set ~4.8GB.

**Measurement protocol (P19's lesson, twice-proven):** A/B on the LIVE full stack only — `python -m
scripts.measure --mic output/_zh_q.wav` (+ `_zh_q_def/why/wx.wav`; `LANGUAGE=zh` live), verify FRESH `[TTFO]`
per run, read `[musetalk sync] hold=` + `[avatar timing] lips start +Xs` from `logs/pipeline.log`; the user's
live eye is the FINAL gate (the probe screen passed two configs the eye rejected). The P19 `[chop]` held%
counter NO LONGER EXISTS (reverted). Gotchas: `MUSETALK_*` knobs reach the avatar server only via a full
relaunch (launcher/`run.ps1`) — the config panel Restart cycles the pipeline ONLY; the avatar server is
single-client (close `/client` tabs before probing); working tree is uncommitted on
`feat/ttfo-first-clause-split`.

## ⭐ Session 2026-07-04 (5th): hop_zh REVERTED to 0 (zh ~4.1→~3.1s); lead<14 REJECTED by the live eye — lead question CLOSED

**Resolved P19's open caveat with a live A/B (Sonnet subagent, full restart dance, fresh-[TTFO]-guarded).**
Fresh baseline first: en median ~3.0s (3.00/3.03/5.72), zh ~4.14s (4.14/4.69/4.05) — handoff decomposition
showed LLM (~0.7s) and TTS first-chunk arrival (~2.1–2.5s) are ~IDENTICAL across languages; **the whole zh
excess was the steady-hold** (en ~0.85s vs zh ~1.9–2.2s), caused by `COSYVOICE_FIRST_HOP_ZH=5`'s small opening
chunk filling the `lead=14` cushion slowly (exactly P19's warning). **A/B result: hop_zh=0 → zh median 3.09s
at lead=14** (hold →~0.8s), smoothness screen clean (freeze ≤147ms, gaps ≤63ms, fps ≥12.0, lips-start
*improved* +0.6–0.9→+0.3–0.4s). **SHIPPED: `run_vllm_server.sh` default is now `:-0`** (hop0/lead14 was
already eye-validated smooth in P19). **Candidate `MUSETALK_LEAD_FRAMES=8` — REJECTED by the live eye
(2026-07-04 evening).** It measured **zh 3.03 / en 2.48 median — the first all-runs-under-3s config** (hold
~0.5s, one live probe turn 2.50s), but the user live-tested every lead below 14: all show delay or avatar
freezes. Lead=14 is FINAL; the probe screen misses what the eye catches (P19's lesson repeated — P22 addendum;
lead=10 skipped, P19 Data 4 shows no gain at hop=0). Remaining TTFO levers (post-lead): the TTS first-chunk
cost + the turn-start hold-spike tail (en outliers 3.3–5.7s = the P20 GPU collision,
hop/lead-independent → P20's stagger/stream-priority levers or the dedicated GPU) + unchanged long-reply lip
drift. Ops note: launch the WSL cosy server from a PERSISTENT background shell (`wsl bash -c "nohup … &"`
gets SIGTERMed with its wrapper); the in-place restart kept the WSL IP.

**Same night, follow-on — zh comma-only first-piece split SHIPPED (`COSYVOICE_FIRST_PIECE_ZH=1`).** Live zh
testing showed the residual zh tail: when the LLM ignores the ≤10-char-opener prompt rule (~30% of turns,
definitional questions) and writes a 19–44-char first sentence, CosyVoice prefills it whole → TTS TTFB ~3.1s →
turn ~4.6–4.8s. The en split never fired for zh (keys on ASCII comma/space). Fix (`first_piece_aggregator.py`,
+47 lines, env-gated, code default 0): flush the turn's FIRST piece at a full-width ，；： only — never a
char-cap (that was the rejected mid-word cutter 天氣預|報) — with a `_ZH_MIN_CHARS=5` CJK-counted guard
(comma before min → wait for the next one; comma-less sentence → identical to before). **A/B (live, fresh-TTFO,
hop0/lead14): long-opener turns 4.78→3.08s / 4.83→3.42s (TTS span −1.2–1.5s, matching the isolated probe
3.07→1.88s); split-fired audio gaps 59–65ms max (no between-clause pause, MIN=5 sufficed); comma-less + en
buckets unchanged (en byte-identity unit-proven).** Unit tests in scratchpad, preflight clean. Note the split
also fires at full-width ：. Test-tooling side effects: edge-tts installed (user pip, test-only) because
Deepgram garbles CosyVoice-synthesized questions + the ElevenLabs key is dead (401); new zh question wavs
`output/_zh_q_def/why/wx.wav`. Synthetic-mic runs can be interruption-contaminated (VAD splits the wav's
internal pause → llm→tts >1.0s marks those turns). All uncommitted; **the user's ear on a couple of live zh
turns is the final gate** (probe gap-screen only). zh state: `LANGUAGE=zh` live, both zh levers now = split
(this) + hop0 (persisted in `run_vllm_server.sh`, default `:-0`). Full write-up: `docs/PROBLEMS-AND-FIXES.md` P23.

**Also same night — phone-loudspeaker fix (`CLIENT_FORCE_SPEAKER=1`, default ON; v2 after the user's first
phone test failed).** Android Chrome switches to "communication" (earpiece) audio routing while the page holds
a live mic → the avatar's voice comes out of the quiet earpiece on phones. Fix = `main.py::
_install_client_speaker_route()`, the sanctioned env-gated `<head>` injection pattern (bundle untouched).
**v2 (the shipped version) hooks `HTMLMediaElement.prototype.play`** — not a DOM sweep, which misses media
elements the bundle never attaches / hides in a shadow root — and tries two routes per element: (1)
`setSinkId()` to the "Speakerphone/Speaker" output (Android Chrome; labels appear only after mic permission,
re-picked on `devicechange`); (2) **iOS fallback** (no `setSinkId` there): pipe the element's MediaStream
through an AudioContext — its output uses the media/playback route (loudspeaker), not the earpiece
'communication' route; the element is muted only AFTER the context is confirmed running so audio can never
vanish (one-shot pointerdown resume covers iOS's gesture rule). Mobile-UA only (desktop/headphones never
touched). **Diagnosable remotely:** the script POSTs each step to `/client/speaker-debug` → `[speaker-debug]`
lines in `pipeline.log` (the handler re-asserts the log sink via `ensure_file_sink` — pipecat's runner tears
all sinks out until a bot session starts, which silently ate pre-connect beacons). The served index now sends
`Cache-Control: no-store` (a phone that cached the pre-patch page misses every injected fix — the likely
reason the user's first test failed). The jitter-buffer + speaker patches share ONE middleware/patch list
(two separate middlewares would shadow each other on the index) — jitter behavior unchanged; both injections
verified served on `/client/`, JS node-checked, beacon round-trip verified. `0` = off. Awaiting the user's
phone re-test (fully close the tab first); uncommitted. `docs/PROBLEMS-AND-FIXES.md` P24.

## ⭐ Session 2026-07-03 (4th): LLM cloud hop → Groq pin, zh short-first-sentence, goal → 3s

**Un-deferred the "LLM cloud-hop variance" item (P19's open question).** Measured the full TTFO budget
with `scripts.measure` (5×/lang, steady): the LLM hop was the single largest component AND its entire
variance/tail. The LLM is pre-warmed on connect, so the measured hop is *pure cloud TTFB*.

**FIX SHIPPED — pin OpenRouter to Groq, ~5 lines, config-shaped.** New knob
`OPENROUTER_PROVIDER_ONLY=Groq` (`pipeline/config.py` + `stages/llm.py`) injects `extra_body.provider.only`
through pipecat's `Settings.extra`; reuses the existing OpenRouter key, empty knob = unpinned Gemini,
fully revertible. **LLM hop halved + tail killed:** zh 1.64→0.80s median (max 3.59→1.44); en 1.07→0.67s.
**BUT end-to-end TTFO only modestly better** — en ~3.95→~3.5s median, zh median ~flat (mean 5.33→4.79s,
worst 6.58→5.25s): the LLM was only ~1/3(en)/~1/6(zh) of the budget; the shared-GPU **TTS (~1.8s) +
steady-hold (~2.0s zh) now dominate**. Real prize = the 7–8s LLM-tail outliers are structurally gone.

**MODEL BASELINE = `meta-llama/llama-4-scout` on Groq** (after a wider model search). It beats the current
choice on every axis: same speed (Groq, non-reasoning, TTFT ~0.6–1.1s), clean Traditional-Chinese that is
*substantive + accurate*, and **~5× cheaper** — $0.11/$0.34 vs llama-3.3-70b's actual Groq price **$0.59/$0.79**
(the "$0.10" seen on `/models` is DeepInfra, the cheapest provider, NOT the pinned Groq endpoint). In a clean
5-question isolated eval: scout gave correct Taiwan-idiomatic answers (台北101 w/ specifics, a real egg recipe);
`llama-3.1-8b` had ERRORS (認主意 nonsense, mislabeled 四四南村, truncation) → rejected; `llama-3.3-70b` good but
terser. Every mid-cost model (gpt-oss-20b/120b, qwen3-32b) is a *reasoning* model → slower; Qwen2.5-72b is only
on non-fast providers + mainland vocab. **GOTCHA:** `pipeline.log` never records a turn's OUTPUT text (only
prior-turn context + token counts) and `measure` runs single-turn — so judge model quality with an ISOLATED
multi-question probe, never the log. Full write-up: `docs/PROBLEMS-AND-FIXES.md` P21.

**Second zh lever — short-first-sentence prompt (`pipeline/config.py` mandarin system prompt).** CosyVoice
prefills the whole first sentence before any audio, and the first-clause split (the en lever) barely fires
for zh; so instructing "第一句話要特別短（十個字以內）" trims the zh TTS first-chunk. Measured: zh TTS hop
1.82→1.67s, zh TTFO ~5.14→~4.34s median (~0.3–0.5s), quality intact. Llama occasionally ignores the ≤10-char
rule on definitional questions (one 18-char straggler).

**Goal tightened `< 8s` → `< 3s`** across all docs + code defaults (`TTFO_TARGET_SECONDS` default, `TtfoMeter`,
`measure.py` display — which also fixed a hardcoded "target 8s" that disagreed with the live 3s meter). Honest
consequence: current en ~3.5s / zh ~4.3s now sit **over** the new bar, so every turn logs `[TTFO OVER]`. The
remaining ~0.5–1.5s is shared-GPU-bound → a **dedicated avatar GPU** is the realistic path to 3s on zh.
**`live` sync is OFF THE TABLE** — the user tested it, the voice leads the lips ~1–2s, rejected (keep `steady`).

**State:** LANGUAGE restored to `zh`. All the above is **NOT git-committed** (held for live A/V human sign-off,
same as the prior TTFO work). Memories: `project-visualllm-llm-groq-pin-ttfo`, `feedback-visualllm-steady-not-live`.

## ⭐ Session 2026-07-03 (3rd): FP8 dead-end + handoff of Lever 1 (GPU priority) & Lever 3 (stagger)

**Symptom re-tackled:** "some turns the avatar's mouth moves before the voice." Root cause (again) = the ONE
shared 16GB GPU: at turn start CosyVoice's opening vocoder burst and MuseTalk's first render segment collide,
the render loses its slice → in `steady` (video-master) the voice is held / the avatar's already-rendered
frames show first. Full write-up: `docs/PROBLEMS-AND-FIXES.md` P20.

**What was tried this session — FP8 quantization of the render UNet (Lever 2a) = REJECTED, PROVEN dead.**
Halving the UNet's GPU math would shorten the starved first segment. Result: **quality flawless (SSIM 0.99997
vs fp16) but 4.5x SLOWER** (fp16 44.4ms → fp8 200.5ms per UNet batch). Cause = **TensorRT 10.13.3.9 cannot
compile FP8 convolution kernels for Blackwell sm_120** (82 skipped FP8 tactics + 30 `NVRTC Compilation
failure`). Ruled out the "modelopt used per-channel weight scales, TRT FP8-conv wants per-tensor" hypothesis
by a decisive **per-tensor tiebreaker** (`trt_quant_fp8.py --per-tensor`): the scale became a scalar `f16[]`
(format fixed) but the failure was **byte-identical** — so it's purely the sm_120 kernel gap, not the format.
Corroborated by NVIDIA/TensorRT issue #4715 (sm_120 Myelin gaps) + the FP8-conv-tactics forum thread. **FP8
stays dead until a newer TRT ships sm_120 FP8-conv kernels; retry then with `--per-tensor`.** Tooling kept:
`local_services/musetalk_server/trt_quant_fp8.py` (modelopt ONNX PTQ, `calibration_method="max"`, real-WAV
calibration, SSIM+speed validation). Env fully rolled back to `logs/_env_snapshot_pre_fp8.txt` (modelopt +
onnxruntime-gpu uninstalled, numpy/onnx restored — verified bit-identical). Detail:
`project-visualllm-fp8-quantization-deadend` memory.

**Live measurement (system-python `scripts.measure`, real driven turn):** an 11s reply rendered at a sustained
12.1fps but the video sat **~2.7s behind the voice** for the whole turn (`[musetalk sync] hold=2.71s`,
`[avatar timing] lips start +2.88s after voice`). This is turn-START latency under **real** CosyVoice
contention — it directly contradicts P16's offline "TRT holds drift flat at +0.36s", which was measured
WITHOUT CosyVoice. Lesson (again, per P19): the shared-GPU cost only shows on the LIVE full stack.

**HANDOFF — the two untried, non-hardware levers (both attack the turn-start collision directly):**

- **Lever 1 — GPU stream priority (let the avatar's render cut the line).** The GPU currently time-slices
  CosyVoice and MuseTalk ~equally during the collision. Mark MuseTalk's render as a **high-priority CUDA
  stream** so its kernels preempt CosyVoice's; the avatar keeps ≥12fps and CosyVoice finishes its audio a few
  ms later (it has huge slack — RTF<1). **Implementation:** in `musetalk_server/app.py`/`trt_runtime.py`, run
  `render_segment` under a high-priority stream — `hp = torch.cuda.Stream(priority=-1)` (lowest number = highest
  priority; check the valid range via `torch.cuda.Stream.priority_range()`), wrap the render body in
  `with torch.cuda.stream(hp): ...` so `trt_runtime`'s `torch.cuda.current_stream()` (used by
  `execute_async_v3`) picks it up, then `hp.synchronize()`. **Honest caveat = the whole risk:** CosyVoice runs
  in **WSL** and MuseTalk in **Windows** — two separate GPU processes/contexts under the Windows **WDDM**
  scheduler, which does NOT strongly honor cross-process CUDA stream priority (CUDA MPS, which would, is
  Linux-only, not on Windows). So priority may not "bite" across the WSL/Windows boundary. **Cheap to test,
  uncertain to work** — measure the live `hold=`/`lips start +` before vs after. If WDDM ignores it, this lever
  is dead and Lever 3 or a dedicated GPU is the path.

- **Lever 3 — stagger the two bursts so they don't overlap.** They only collide because both peak at the same
  instant; CosyVoice's burst is **short + front-loaded** (it sprints the opening audio faster than real-time,
  then quiets). Two implementation options: **(a)** throttle CosyVoice's generation nearer real-time at turn
  start so it doesn't front-load the GPU — lever on `COSYVOICE_PACE_RATE` (currently 1.3, in the cosyvoice
  repo's server), possibly a dedicated turn-start throttle, so it spreads GPU use and leaves the avatar's first
  render room; **(b)** on the MuseTalk client (`musetalk_video.py`), delay feeding the first render segment by
  ~150–250ms so CosyVoice's opening burst clears first, then render against a quieter GPU (trade: adds that
  fixed delay to lip-start). **(a) is lower-risk** (a TTS-side throttle, no sync surgery) and preferred first.
  Measure the same live `hold=` delta; watch that TTS TTFO doesn't regress past the 8s target.

  **Measurement protocol for BOTH levers (do not repeat P19's mistake):** A/B on the **LIVE full stack** with
  real CosyVoice — `python -m scripts.measure --mic output/q_long.wav` — and read `[musetalk sync] hold=` +
  `[avatar timing] lips start +Xs` from `logs/pipeline.log` (offline GPU-hog tests LIE — not bursty like
  CosyVoice). The human eye is the final gate. A config only ships if it cuts the live hold AND stays smooth.

  **Also on the table if both levers fail:** INT8 quantization (same idea as FP8 but INT8-conv kernels DO exist
  on sm_120 — the honest technical redo, quality gamble); a turn-start "about-to-speak" gesture to MASK the lag
  (perception fix, low risk, `MUSETALK_IDLE_MOTION` machinery); `MUSETALK_SYNC_MODE=live` (voice always first,
  lips trail ~0.75s — the direct symptom cure the user previously rejected); a bounded steady voice-hold
  (`MUSETALK_SYNC_MAX_HOLD_S`, best-of-both but risky sync-code change); or the structural fix, a dedicated
  avatar GPU. **The shared GPU has now been hit from 5 angles (FP8, hop/lead, GPU-composite, TRT, live
  measurement) — a dedicated GPU remains the only guaranteed cure.**

> **⭐ Baseline (2026-07-02) — the known-good config to return to:**
> **Avatar:** `MUSETALK_TRT=1` (TensorRT render, merged to `main`), `MUSETALK_GPU_COMPOSITE=1` (GPU per-frame
> blend, opt-in; needs TRT), `MUSETALK_SYNC_MODE=steady`, `MUSETALK_FPS=12`, `MUSETALK_SIZE=256`,
> `MUSETALK_LEAD_FRAMES=14`, `MUSETALK_END_TAIL_FRAMES=0`, `MUSETALK_CLOSE_FADE_FRAMES=5`, `MUSETALK_IDLE_MOTION=0`.
> **LLM = cloud** `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1` + `OPENROUTER_MODEL=google/gemini-2.5-flash-lite`
> (do NOT use a *thinking* model like `qwen3.5:4b` — it returns empty `content`; `OPENROUTER_REASONING_EFFORT`
> is a dead knob the pipeline never reads).
> **TTS = `TTS_PROVIDER=cosyvoice`** (WSL vLLM, `COSYVOICE_URL` = WSL IP). The CosyVoice repo
> (`E:\Claude\cosyvoice-local-tts`) now bakes in two zh fixes (in code, not `.env`): **(1) RAS restored in the
> vLLM sampler** (`CosyVoice/cosyvoice/vllm/ras_logits_processor.py` + `top_p=0.8` in `llm.py`) — stops the
> intermittent silence-loop that made zh "halting"; **(2) the "pro" AI-assistant reference voice**
> (`CosyVoice/asset/pro_ref.wav`, the default in `tts_engine.py`) — naturally fluid zh (~1 pause/sentence,
> ~64% voiced ≈ English). The zh pause-trimmer is OFF by default (`COSYVOICE_SILENCE_CAP_S=0`; not needed with
> the pro voice). One-click: **`Run VisualLLm.exe`**.
> **TTFO — per-language levers (2026-07-03):**
> **English → the split** `COSYVOICE_FIRST_PIECE=1` (MIN=18/MAX=32, `.env`) — emit a short opening clause to
> TTS first, then normal sentences. en's long sentences make the early-clause start worth **TTFO ~4.6s→~3.2s**,
> smooth (gap ~55ms). Code: `local_services/first_piece_aggregator.py` (gated; `=0` = plain aggregation).
> **Chinese → FIRST_HOP** `COSYVOICE_FIRST_HOP_ZH=5` (cosyvoice repo's `run_vllm_server.sh`), applied to
> **Chinese ONLY** (`tts_engine.py::_apply_first_hop`, per-request by `is_cjk`) — emits the first audio after
> fewer speech tokens, capping zh's bigger opening → zh first-chunk **~2.5s→~1.8s**, whole natural sentences,
> no avatar starvation (TensorRT holds ≥12fps; the old pre-TRT "starves" verdict is void *for zh*).
> **English is forced to `hop=0`** (`COSYVOICE_FIRST_HOP_EN`, default 0): the earlier *global* hop=5 was
> measured pushing en **lip-start lag ~0.70s→~1.95s** (its turn-start vocoder burst starves MuseTalk's
> first-frame render, which TRT's mid-turn headroom doesn't cover) for no en benefit. Verified 2026-07-03:
> isolated TTFB en 3.15s / zh 2.00s; live en lip-start back to +0.70s (flat +0.62s offset all turn). They
> coexist: the split fires on long en sentences, hop=5 only bites short zh ones. (Splitting zh was tried and
> rejected — it cuts mid-word, 天氣預|報, with no TTFO win.) The LLM cloud-hop variance now dominates
> worst-case end-to-end TTFO in both languages (a separate, deferred lever).
> **⚠️ 2026-07-03 re-measurement caveat (see P19):** a later live sweep found `FIRST_HOP` HURTS **live**
> TTFO even for zh at the default `lead=14` (zh hop=5 → live TTFO ~5.78s vs hop=0 ~3.68s) — the earlier
> "zh→hop=5 helps" was measured on isolated first-chunk TTFB + fps, which miss the synced-start fill delay.
> The shipped `COSYVOICE_FIRST_HOP_ZH=5` is **worth re-evaluating** (data says hop=0 is better for live zh
> TTFO too), but zh was NOT choppiness- or human-A/V-validated this session, so it is left as-is pending a
> live zh check. English is unaffected (already `hop=0`).

## ⭐ Session 2026-07-03 (2nd): TTFO tuning sweep — NO WIN, baseline `hop=0, lead=14` stands

Swept `COSYVOICE_FIRST_HOP` × `MUSETALK_LEAD_FRAMES` looking for a lower TTFO. **Negative result — shipped
nothing.** Full method + all data tables in **`docs/PROBLEMS-AND-FIXES.md` P19**; the essentials:

- **`LEAD_FRAMES` is the synced-start delay** (the pump holds the voice until `lead_frames` frames are
  queued), and it is a **mid-turn shock absorber**. `FIRST_HOP<25` = a smaller opening TTS chunk.
- **The isolated TTFB win from a low hop is REAL but erased LIVE.** In steady, a smaller opening chunk fills
  the `lead=14` cushion slower → the voice-start is *delayed*; and hop's extra small vocoder bursts contend
  with MuseTalk on the shared GPU. Live zh TTFO: hop=5 ~5.78s vs hop=0 ~3.68s (WORSE). Isolated TTFB
  saturates by hop≈3 (en 2.62s→1.86s, zh 3.04s→1.84s).
- **The "low hop starves the avatar" (P15) is a `lead=14` artifact** — small chunk vs big cushion; the
  synced-start `delay` has a cliff between lead 10 and 8 (drops to ~0.4–0.6s at lead≤8 for every hop).
- **BUT lowering the cushion to recover the win re-introduces choppiness.** Live server-side choppiness
  metric (`[chop]` held%, n=4): baseline hop0/lead14 = **17.6%** (smooth); **hop5/lead6 = 36.5% (CHOPPY**,
  confirmed by the user's eye); hop0/lead6 = 17.2%, hop5/lead14 = 16.9%, hop0/lead10 = 17.3% (all smooth).
  **Only the COMBINATION hop5+lead6 is choppy** — hop's bursty contention + a thin cushion → mid-turn
  underflow. Either knob alone is fine.
- **An offline contention "PASS" was misleading:** `_drive_frames`+`_gpu_contention_hog.py` (100% GPU, 13.6s
  reply) showed `lead=6`≡`lead=14` (no underflow) — but a *steady* matmul hog isn't *bursty* like CosyVoice
  and ran without it. **Lesson: A/B on the LIVE full stack + human eye; measure choppiness server-side**
  (WebRTC duplicate-detection and `freeze_ms` both failed to see it).
- **"Avatar starts before the voice" is normal steady behavior, not a hop/lead bug** — audio arrives ~2s
  ahead of video (`[musetalk sync] hold=2.07s, audio 4.2s, video 2.2s`), steady is video-master so it holds
  the voice for the render; when the render lags, lips play first. **The lever for that is
  `MUSETALK_SYNC_MODE` (steady↔live), NOT hop/lead.**
- **State:** all temporary instrumentation reverted (two `/debug` endpoints + a `[chop]` counter);
  `musetalk_server/app.py` `git`-clean; `.env` unchanged; stack restored to the clean baseline. A genuine
  TTFO win needs a **dedicated avatar GPU**, not a settings tweak.

## ⭐ Session 2026-07-02: Chinese TTS fixed (RAS restored + fluid "pro" voice)

**Symptom:** zh sounded broken ("like autism speaking" — long unnatural pauses) and the avatar kept
moving after the voice ended; **English was perfect**. Both were ONE TTS bug, not the avatar/steady/GPU.

**Root cause (proven offline via `/tts/stream`, concatenated-PCM analysis):** CosyVoice2's LLM runs on
**vLLM** here (the latency fix), and the vLLM decode path does its own sampling — it **dropped CosyVoice's
repetition-aware sampling (RAS)**. RAS (native PyTorch path, `cosyvoice/utils/common.py`) bans a speech
token that just recurred in the last ~10 tokens; without it the model intermittently **loops on the silence
token** — the SAME short zh sentence came out ~4s clean one run and **~12s with ~5s of dead silence** the
next (~40% of zh runs). That silence *is* both symptoms: the pauses = "halting" voice; MuseTalk renders
frames for the whole 12s (idle mouth) while the words are ~4s = "avatar keeps moving". zh uses
`inference_zero_shot` (denser tokens) and hits it; en uses `cross_lingual` and doesn't.

**Fix 1 — restore RAS in the vLLM sampler (the real root-cause fix, in `E:\Claude\cosyvoice-local-tts`):**
`CosyVoice/cosyvoice/vllm/ras_logits_processor.py` = a vLLM V1 per-request logits processor that bans any
token seen in the last `COSYVOICE_RAS_WIN` (=10) OUTPUT tokens, registered via
`EngineArgs(logits_processors=[...])` in `cli/model.py`, plus `top_p=0.8` in `llm.py` to match RAS's nucleus.
**DEAD END:** vLLM's own `repetition_penalty`/`frequency_penalty` build a prompt-token bincount, but CosyVoice
feeds `prompt_embeds` (no prompt token ids) → CUDA `ScatterGatherKernel index out of bounds` device-side
assert that kills the engine. Verified: **48 zh runs, 0 degenerate** (was ~40%); tighter + lower-latency
than an earlier output-guard/retry workaround (which was reverted).

**Fix 2 — a naturally fluid "pro" reference voice (the actual reason zh was choppy-vs-en):** after the loop
was fixed, zh still felt choppy/slow (~57% voiced / ~3.8 pauses/sentence vs en's ~65% / ~2.5). Ruled out the
`speed` knob and the `cross_lingual` path (both barely helped). The lever was the **reference clip** —
`zero_shot` clones its rhythm. Swapped the gappy "weather" clip for the **MOSS "pro" AI-assistant voice**
(`assets/moss_pro_ref.wav` → copied to `CosyVoice/asset/pro_ref.wav`, transcribed via Deepgram, now the
default in `tts_engine.py`): zh → **~64% voiced / ~1 pause/sentence** (fewer than English), no trimming
needed. (An interim streaming pause-trimmer `_squeeze_silence` / `COSYVOICE_SILENCE_CAP_S` is now **OFF by
default** — kept as an optional knob for gappy voices.) **Correction to an earlier claim:** the choppiness
was NOT "inherent to the model" (that was premature) — it was the reference voice, and CosyVoice2 is in fact
Chinese-first and fine at zh.

**Status:** live + verified offline; changes are in the `cosyvoice-local-tts` + nested `CosyVoice` repos
(pro voice clip committed there), **not yet git-committed**. Full detail: `docs/PROBLEMS-AND-FIXES.md` P18 +
the `project-visualllm-zh-silence-loop-fix` memory. Avatar/LLM baseline unchanged.

## ⭐ Session 2026-07-01: TensorRT avatar = new baseline (long-turn drift fixed)

**Root-caused "avatar frames != audio frames" + "sync drift on long turns" by driving the MuseTalk
server with real WAVs at prod fps (offline, no CosyVoice/WebRTC).** Findings: the RENDERED lip-frame
count (server `video_clock`) DOES equal `audio*fps` (±1, the P9/P10 ceil pad is correct) — the "extra"
frames are the pump's HELD/duplicate frames that keep the WebRTC track continuous whenever render dips
below fps. Per 8-frame segment on the PyTorch path: gpu 259ms + composite ~120ms ≈ 389ms vs the 667ms
real-time budget @12fps (~1.7x headroom), so ALONE it barely drifts (fixed +0.36s startup offset any
length). The drift becomes length-scaling ONLY when render drops below 12fps — proven with a GPU compute
hog (100% util, CosyVoice stand-in): PyTorch drifted `+0.37s(2.9s) -> +1.35s(5.5s) -> +3.94s(13.6s)`,
render ~9fps. **So the long-turn drift is shared-GPU contention, NOT the frame math.**

**Fix = TensorRT, now merged to `main` and default (`MUSETALK_TRT=1`).** Ported the prebuilt UNet+VAE TRT
engines (were on `feat/offline-stt-sensevoice`). Measured: gpu 259->168ms, composite ~120->78ms,
total/seg 389->~255ms (~1.5x; headroom 1.7x->2.6x). **Under the SAME 100% contention that drifted the
PyTorch path +3.94s on the 13.6s reply, TRT holds drift FLAT at +0.36s at every length** (held frames
50->4). Off unless `MUSETALK_TRT=1`; any engine-load failure falls back to PyTorch. Engines are
GPU/driver-specific (~1.75GB, gitignored) — rebuild with `local_services/musetalk_server/trt_build.py`.
Next cheap lever (no 2nd GPU): the composite is CPU PIL blending (~31% of render even with TRT) — move it
to GPU. Structural fix remains a dedicated avatar GPU.

**Follow-on same session (GPU composite + engine-build CLI).** Took the two cheap levers off the
P16 handoff list. (1) **GPU composite (`MUSETALK_GPU_COMPOSITE=1`, P17):** moved the per-frame
mask-blend + downscale off the CPU (PIL/cv2) onto the GPU in torch. **Benchmarked OFF vs ON** (clean
no-contention drive, TRT on both, 13.56s reply): per 8-frame seg gpu ~170ms unchanged + composite
**~73ms → ~11ms** (6.6×) → total **246 → 182ms**, i.e. **−26% render, ceiling ~33 → ~44 fps**.
**Pixel-identical** (SSIM 1.0, ≤1 LSB, verified vs CPU on smooth/random/checkerboard content).
**HONEST caveat:** at the production 12fps this does **NOT** change A/V drift — both configs hold drift
flat (+0.69s, 7 held frames) at all lengths **even under a verified 100% GPU-contention hog**, because
TRT alone already keeps render ≥12fps. The win is **reserve** (render ceiling 33→44fps + a freed CPU for
STT/pipecat/LLM-streaming), which the offline render-isolation test can't see — the **live call (#1)** is
the judge of the CPU-relief benefit. Opt-in, code default off, needs TRT (uses the VAE GPU tensor
directly); this box's `.env`=1, `run.ps1` propagates it; CPU fallback if a crop_box runs off-frame.
(2) **One-command engine-build CLI:** `python -m local_services.musetalk_server.trt_build` now wraps the
4 export/build one-liners (forces the PyTorch path during capture, derives seq-len + max batch from the
model). Both validated offline; **the live A/V judgement (handoff #1) is still the open item.**

> **See `WORKFLOW.md`** for the full end-to-end system workflow (the processes, the turn
> flow, the avatar wire contract, running locally + remote, config reference).
> **See `docs/PROBLEMS-AND-FIXES.md`** for the catalogue of bugs found + how each was fixed.

## The stack

| Stage | Service | Where |
|-------|---------|-------|
| VAD | Silero (local) | pipeline |
| STT | Deepgram nova-2 (`en`/`zh`/`th` by `LANGUAGE`) | cloud |
| LLM | `LLM_PROVIDER=openrouter` (OpenAI-compatible — **cloud OR local Ollama** by `OPENROUTER_BASE_URL`) or `weather_chain` (Chinese weather bot) | cloud / local / remote chain |
| TTS | **CosyVoice2-0.5B** local streaming (default), or **MOSS-TTS-Realtime** (`TTS_PROVIDER=moss`) | `:8001` cosy (WSL) / `:8003` moss (WSL) |
| Avatar | **MuseTalk** local mouth-region talking-head (female portrait) | `:8002`, `musetalk` conda env |
| Config | **Web config panel** — edit `.env` + restart pipeline from the browser | `:7870` (`:8444` over Tailscale) |

WebRTC → browser at `http://localhost:7860/client/`. Goal: time-to-first-output **< 3 s**.

TTS providers (`cosyvoice` default · `moss` · `elevenlabs` · `deepgram`) and LLM providers
(`openrouter` · `weather_chain`) are deliberate **single-provider switches** via `.env`, not
multi-provider branching. **Easiest way to change any of this: the config panel (`:7870`).**

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
  `scripts/_drive_frames.py` — frames-vs-audio + render-fps probe (drives :8002 with a WAV, no
  CosyVoice/pipeline/WebRTC); `scripts/_gpu_contention_hog.py` — GPU compute hog to reproduce the
  shared-GPU drift offline. Together they produced the P16 numbers.
- `archive/` — kept-out-of-tree regression tests.

## Handoff / next (as of 2026-07-01, end of the TRT-baseline session)

**Shipped + pushed to `origin/main` (`226497d`):** TensorRT avatar is the baseline (`MUSETALK_TRT=1`),
fixing the shared-GPU long-turn A/V drift; LLM = cloud `google/gemini-2.5-flash-lite`; all stack docs
+ P16 propagated. Commits: `dfa1552` (TRT code) → `bd1d765` (merge) → `d198ea1`/`30f93bd`/`226497d` (docs).

**Verified this session:** `_drive_frames.py` + `_gpu_contention_hog.py` proved rendered lip frames
= `audio*fps` (P9/P10 correct); the drift is contention-driven (`drift ≈ audio_len*(1−render_fps/fps)`),
and TRT holds it flat (+3.94s→+0.36s at 100% contention on a 13.6s reply). Live stack was left running
(cloud LLM + TRT); TRT load confirmed in `logs/musetalk.err.log` ("TRT engines loaded").

**✅ Done in the follow-on session (2026-07-01, NOT yet committed — working tree only):**
- **#2 Composite-on-GPU** — `MUSETALK_GPU_COMPOSITE=1` (P17). ~68ms→~4–15ms/seg, pixel-identical
  (SSIM 1.0). Opt-in, needs TRT, CPU fallback. Code `app.py` (`_init_gpu_composite` + `_composite_gpu`),
  wired into `.env` + `run.ps1`.
- **#3 One-command engine build** — `python -m local_services.musetalk_server.trt_build` (`build_all`
  in `trt_build.py`); SETUP.md updated.
- **#6 `.superpowers` hygiene** — already safe: `.superpowers/sdd/.gitignore` is `*`, so the
  forgetting-benchmark workspace is self-ignoring and cannot leak to public `main` via `git add`. (The
  only tracked "benchmark" is `tts/cosyvoice-server/benchmark.py` = the CosyVoice latency benchmark.)

**Not yet done / open, in priority order:**
1. **Human A/V judgement on a real call** — open `/client/` (mic + headphones), speak a long
   multi-sentence turn, confirm the lips stay locked to the voice end-to-end with TRT + GPU composite on.
   The offline proof used a synthetic hog, not the real CosyVoice burst pattern. **This gates promoting
   `MUSETALK_GPU_COMPOSITE` from opt-in.** _(Stack was left LIVE with GPU composite on: MuseTalk :8002 +
   pipeline :7860 + CosyVoice :8001 all up — test it directly.)_
2. **Commit the follow-on work** — the #2/#3 changes above are in the working tree only, not committed.
3. **Dedicated avatar GPU** — the structural fix; removes contention entirely (avatar working set ~4.8GB).
4. **Confirm the felt CosyVoice-vLLM latency** on a live remote call; Chinese zh-TTS delay (P15) still open.

**To run cold:** double-click `Run VisualLLm.exe` (brings up WSL CosyVoice → MuseTalk+TRT → pipeline →
`/client/`). If the bot goes silent after a `wsl --shutdown`, the WSL IP changed — update `COSYVOICE_URL`
(`wsl hostname -I`).
