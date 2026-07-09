# CUDA graphs without the drawback — diagnose then fix

**Date:** 2026-07-05
**Context:** P27 enabled vLLM CUDA graphs (`COSYVOICE_VLLM_EAGER=0`) → TTS first-chunk
avg ~2.0→~0.85s, but P31 reverted them: live per-turn latency became **inconsistent**
(fast, fast, SPIKE, fast). Goal: keep the speed without the spikes — or prove we can't.

## The two-mechanism split (the crux)

The "graphs = inconsistent" verdict bundles two mechanisms that were **never isolated**:

- **(A) Runtime capture / JIT spikes on novel shapes.** A CUDA graph is captured per
  input shape. If a turn hits a shape not captured at startup, the GPU captures it
  *mid-turn* → that turn spikes. **Fixable:** pre-capture every latency-critical shape
  at startup.
- **(B) Rigid graph replay colliding with MuseTalk on the shared 16GB card.** Replayed
  graphs issue kernels rigidly; eager issues them dynamically so the scheduler interleaves
  around MuseTalk's concurrent TRT render. **Structural:** only a dedicated GPU / MPS fixes it.

Smoking gun for (A): the server warms up exactly **one** shape at boot
(`app.py:48`, `"Hello, warming up."`). Every other opener length is cold with graphs on.

Key lever we exploit: TTFO only cares about the **first chunk**, whose input is **bounded**
by `COSYVOICE_FIRST_PIECE` (en 18–32 chars; zh comma-split ≥5 CJK). The first-chunk shape
space is small and known → exhaustively warmable.

## Part 1 — Diagnostic (isolate A vs B)

Measure **isolated** first-chunk TTFB variance (no MuseTalk, no pipeline; POST
`/tts/stream` directly at the WSL IP — localhost relay fakes the number). Report median,
**max, and stddev** (variance is the point). Drive varied opener lengths across the
FIRST_PIECE band, each repeated N times so first-sighting vs repeat is visible.

| Config | Graphs | Warmup | Question answered |
|---|---|---|---|
| 3 (baseline) | OFF (eager) | current | Steady-but-slow baseline to beat |
| 1 | ON | current (1 shape) | Is variance present *isolated*, from cold shapes? |
| 2 | ON | **band warmup** (fix) | Does exhaustive warmup collapse the variance? |

Confirmation run: Config 2 **with MuseTalk on** → does contention (B) reintroduce spikes?

**Decision rule**
- Config 1 spiky (esp. on first-sighting of each length) **and** Config 2 tight → **(A)**,
  warmup fixes it → ship + hand user a live test.
- Config 1 already tight isolated → live jank was **(B) contention** → warmup can't help →
  honest negative; graphs stay off (needs dedicated GPU/MPS).

## Part 2 — The fix (only if A)

Extend the boot warmup (`app.py:48`) from one string to a set spanning the opener band
(a few en lengths across 18–32 chars + a couple zh comma-split openers) so vLLM
captures/compiles every first-chunk shape at startup. Re-enable graphs by default
(`COSYVOICE_VLLM_EAGER=0`). Cost: +boot time only (launcher `/health` wait tolerates it).
All env-gated and revertible.

## Part 3 — Verdict division

I prove isolated variance (max/stddev). **User** delivers the final "is it steady live"
verdict on a call — the whole P19/P27 history says the probe passes what the eye rejects.
No "it's fixed" claim off numbers alone.

## RESULTS (2026-07-05, executed)

Built `_ttfb_variance.py` (first-chunk TTFB over `/tts/stream`, hits WSL IP, varied
opener lengths × N rounds, reports median/max/stddev + round1-cold vs warm penalty).

| Scenario | graphs (med/max/std) | eager (med/max/std) |
|---|---|---|
| Isolated, fresh graphs (cold shapes, r1) | 1.12 / 1.76 / 0.32 | (n/a) |
| Continuous GPU hog (N=4096) | 2.75 / 3.57 / 0.53 | 3.21 / 3.98 / 0.63 |
| **Real MuseTalk TRT render** (96 samp) | **1.29 / 2.23 / 0.37** | 1.94 / 3.43 / 0.64 |

- **(A) cold-shape capture DISPROVEN:** fresh graphs server, round-1 novel shapes == warm
  rounds (cold penalty -0.03s). No `<-- COLD SPIKE` flags in any run. The single-shape boot
  warmup was NOT leaving a runtime-capture hole in TTS TTFB.
- **(B) contention DISPROVEN (opposite of the claim):** graphs are FASTER and LOWER-variance
  than eager under both a continuous hog and real MuseTalk bursty render. The `max` across
  96 graphs samples (2.23s) is just the longest opener's normal cost — no spike event.
- **Band-warmup fix NOT built** — it targeted mechanism (A), which doesn't exist.

**Conclusion:** the P31 "graphs = inconsistent" drawback is NOT in the TTS layer; graphs are
strictly the better TTS baseline. Any residual live inconsistency can only be in **steady-mode
perceived A/V timing** — a *systematic* shift from the ~0.6s-faster first chunk (like the
hop=0/P22 story: faster first chunk fills the `MUSETALK_LEAD_FRAMES=14` cushion differently),
which is **tunable via lead frames**, NOT a random spike. Default flipped back to
`COSYVOICE_VLLM_EAGER=0`; **pending a fresh live-eye A/B** (the arbiter my probe can't be).

## Scope / files
- `E:\Claude\cosyvoice-local-tts\app.py` — boot warmup list (fix).
- `E:\Claude\cosyvoice-local-tts\run_vllm_server.sh` — `EAGER` default (fix).
- New diagnostic probe (TTFB variance over `/tts/stream`).
- Everything revertible; changes live in the separate cosyvoice repo.
