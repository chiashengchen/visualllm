# Design: true per-stage latency to the user's ear

_Date: 2026-07-06 · Branch: feat/ttfo-first-clause-split_

## Goal

Make the measure system report the **actual latency of each part of the system, all the way
to the browser output the user hears** — as a per-stage waterfall that sums to a true
end-to-end number, not just the server-side `[TTFO]` (which stops at
`BotStartedSpeakingFrame`, i.e. when the pipeline *starts pushing* audio, missing the entire
last mile: transport pacing, steady lead-hold, WebRTC encode, network, jitter buffer, and
speaker playout).

## The core problem & the fix

Today two clocks are never joined:

- Server-side stage anchors come from `pipeline.log` (a `datetime` per line).
- Client-side arrival timings come from the headless probe's `time.time()` (`connect_t`,
  `vwall`, `awall`).

They are never stitched, so the last mile is invisible and no per-stage numbers add up to a
real end-to-end total.

**Fix:** the probe and the pipeline run on the **same machine → the same wall clock**. Convert
the log's `t0` (user-stopped) to epoch with `t0.timestamp()` (naive-local → epoch, matching
`time.time()`), and every client arrival becomes a real offset from t0. This single stitch is
what unlocks the whole waterfall. (DST/clock-change mid-run is the only theoretical risk —
negligible for a ~40s run.)

## Architecture — one report, two last-mile sources

Mirrors the existing dual-source `lip_offset` pattern (`offline` vs `webrtc`, labeled in the
report). The last-mile row is filled by whichever source ran; both feed the same waterfall.

### Source 1 — Headless (always on, automated)

`measure.py` already drives the turn (plays a mic wav) and receives the bot A/V over a headless
`aiortc` connection. Changes:

- Extend the **audio pump to record `(epoch, rms)` per received audio frame** (currently it
  records only `time.time()`). `track.recv()` returns an `av.AudioFrame`; `to_ndarray()` →
  int16 samples → RMS.
- **Answer-onset detection:** the first energetic audio frame with `epoch > t0_epoch + guard`.
  An energy threshold (reuse `_webrtc_probe`'s `> 0.18 * max` idiom over the post-t0 window)
  skips both the greeting (well before t0) and inter-turn silence. `guard ≈ 0.15s`, require a
  short sustained run so a single spike doesn't trigger.
- Result: the true *"client received first answer audio +N s after the user stopped"* — covers
  transport output pacing + steady lead-hold + WebRTC encode + loopback network + depacketize.

Honest caveat: the probe is `aiortc`, not Chrome — its depacketize/buffer differs slightly from
a real browser. It measures **network arrival**, not speaker playout; that final segment is
Source 2's job (or an estimate).

### Source 2 — Real browser (opt-in truth layer)

A small env-gated `<head>` injector `_install_client_playout_probe()`
(`CLIENT_PLAYOUT_PROBE`, default OFF — measurement scaffolding, same convention as the av-stats
monitor) attaches an `AnalyserNode` to the inbound bot **audio** track (grab the track via the
same `RTCPeerConnection` `track` event the jitter-buffer patch uses; build
`AudioContext → MediaStreamSource → AnalyserNode`), polls RMS, and beacons **one**
`[client-playout] audio-onset t=<Date.now()>` the first time RMS crosses threshold after
silence, then **re-arms per turn** (disarm on beacon, re-arm after a silence gap). POSTs to a
new `/client/playout` endpoint handled in the existing patch middleware (with
`ensure_file_sink("pipeline")`, exactly like `/client/av-stats`).

This captures the genuine **to-the-ear** moment including the browser jitter buffer, decode, and
the phone loudspeaker route. Because the browser is single-client, this cannot run at the same
time as the headless probe; so `measure.py --from-browser` runs **parse-only** (no probe): it
reads the last turn's server anchors + the `[client-playout]` beacon from `pipeline.log`.

Clock: same-box localhost → `Date.now()` epoch is directly comparable to the log t0 epoch. A
remote/Tailscale browser is a different clock; that case is **labeled low-confidence** (a proper
offset handshake is out of scope).

## Deliverable: a waterfall that sums to the end-to-end

Each Δ traces to exactly one measured anchor; the rows add up from t0 to "user hears":

```
STAGE (from t0 = user stopped)        Δ this stage   cumulative     source
STT finalize -> LLM                    0.02s          0.02s          log
LLM first token                        0.68s          0.70s          log (TTFB)
LLM -> TTS (sentence-1 flush)          0.35s          1.05s          log
TTS synth first chunk                  1.40s          2.45s          log (TTFB)
TTS -> bot-start (steady lead-hold)    0.30s          2.75s          log ([TTFO])
-- server boundary --------------------------------------------------------
Transport + encode + network           0.22s          2.97s          headless probe   <- NEW
Browser jitter + decode + playout      0.15s          3.12s          beacon | est.    <- NEW
-------------------------------------------------------------------
END-TO-END, user hears                 3.12s
```

Stage anchor mapping (all pipeline-clock offsets from t0 except the last two):

| Stage | Start anchor | End anchor |
|-------|--------------|------------|
| STT finalize -> LLM | t0 | LLM receives (~0, pre-warmed) |
| LLM first token | t0 | `llm_ttfb` offset |
| LLM -> TTS flush | `llm_ttfb` offset | `sentences[0]` offset |
| TTS synth first chunk | `sentences[0]` offset | `tts_ttfb[0]` offset |
| TTS -> bot-start (steady hold) | `tts_ttfb[0]` offset | `bot_started` ([TTFO] line) |
| Transport + network | `bot_started` | client answer-audio arrival (Source 1) |
| Browser playout | client arrival | `[client-playout]` onset (Source 2) or estimate |

**Graceful degradation:** a stage whose log anchor is missing is marked `unknown` (row shown,
not faked, and flagged as breaking the sum) rather than silently dropped. The browser row falls
back to an **estimate** — `arrival + measured jitterBufferDelay` if av-stats data is present,
else `CLIENT_JITTER_BUFFER_MS` — labeled `est.` vs `measured`.

## Components touched (surgical)

1. **`scripts/measure.py`** — epoch clock-stitch (`t0.timestamp()`); audio pump records
   `(epoch, rms)`; `answer_onset_epoch()` detector; `build_waterfall(turn, onset)`; new
   client-arrival event + handoff + metric rows; `--from-browser` parse-only mode; dual-source
   last-mile selection (probe vs beacon, labeled).
2. **`pipeline/main.py`** — `_install_client_playout_probe()` injector + a `/client/playout`
   POST branch in `_inject_client_patches`; wired in `__main__` alongside the other injectors.
   Prebuilt bundle untouched, env-gated, default OFF, import-guarded like its siblings.
3. **`docs/workflow-timeline.html`** — a new waterfall panel + client-arrival lane/metric rows.
   `docs/measure_data.js` (`window.MEASURE = {...}`) stays auto-generated by `measure.py`, now
   carrying a `waterfall` array.
4. **`pipeline/metrics.py` — untouched.** `TtfoMeter` stays the server-side acceptance metric;
   end-to-end-to-ear is a measure.py-side derived number, so the live pipeline is not disturbed.

## Testing

- **Pure-function unit tests (no network/GPU):** synthetic `(t, rms)` arrays + a synthetic log
  snippet →
  - onset detector picks the first sustained energetic frame after t0 (ignores a pre-t0
    greeting burst and a single post-t0 spike);
  - clock-stitch: a known `t0` datetime + known arrival epoch → expected offset;
  - `build_waterfall`: Δs sum to the cumulative end-to-end; a missing anchor yields an
    `unknown` row, not a wrong sum.
- **Live integration:** `python -m scripts.measure --mic output/_zh_q.wav` on the full stack →
  the new transport Δ is positive and small (tens–hundreds of ms), the waterfall total is
  within reason of `[TTFO] + last-mile`. Then a real-browser turn with `CLIENT_PLAYOUT_PROBE=1`
  + `python -m scripts.measure --from-browser` → the `[client-playout]` beacon parses and the
  playout row fills as `measured`.
- `python -m scripts.preflight` stays green (the new injector is guarded like the others).

## Non-goals / YAGNI

- No change to the live TTFO metric or the pipeline's frame handling.
- No remote-clock offset handshake (remote browser playout is labeled low-confidence).
- No new video last-mile metric beyond the existing startup + lip-offset (the audio path is what
  defines "user hears"; video sync is already covered by `lip_offset` + av-stats skew).
