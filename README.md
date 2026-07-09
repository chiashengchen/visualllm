# VisualLLm — Speech → LLM → Talking-Head Avatar

A real-time conversational system: **speak to it, and a photoreal avatar speaks
back** (lip-synced audio + video). Multi-turn, streaming end-to-end.

```
speech → STT → LLM → TTS → lip-sync avatar → audio+video out
```

**Goal:** time-to-first-output (you stop speaking → avatar starts responding)
**< 3 seconds**.

The whole system streams: as soon as the LLM emits its first sentence it flows
to TTS → first audio chunk → the avatar starts talking. We never wait for a
stage to fully finish.

---

## Architecture

Built on **[Pipecat](https://github.com/pipecat-ai/pipecat)** — it wires every
stage with streaming + barge-in built in. One pure stack chosen by `.env`; each
stage is a thin single-provider factory with deliberate fallback switches (not
multi-provider branching). Core knobs: `LANGUAGE` (en/zh/th), `TTFO_TARGET_SECONDS`,
`TTS_PROVIDER` (`cosyvoice`|`elevenlabs`|`deepgram`), `MUSETALK_SYNC_MODE`
(`steady`|`live`), and `WEBRTC_ICE_SUBNET` (pin ICE to Tailscale for the remote mic).

| Stage | Service |
|-------|---------|
| VAD / turn-taking | Silero (local) |
| STT   | Deepgram (nova-2; `en-US` / `zh-TW` / `th` by `LANGUAGE`) — cloud |
| LLM   | OpenRouter (any model via `OPENROUTER_MODEL`) — cloud |
| TTS   | **CosyVoice2-0.5B**, local streaming server, female zero-shot voice — **runs on vLLM in WSL** (first-chunk latency ~1.1s; the Windows PyTorch server is the fallback). ElevenLabs / Deepgram Aura are cloud fallbacks via `TTS_PROVIDER` |
| Avatar| **MuseTalk** — local mouth-region lip-sync server on the GPU (5060 Ti), female portrait, **TensorRT render** (`MUSETALK_TRT=1`, default) |
| Transport | WebRTC → browser at `/client/` |

```
pipeline/
  main.py            pipeline assembly + dev runner
  config.py          keys + the en/zh/th switch (single source of truth)
  metrics.py         TtfoMeter — measures time-to-first-output
  stages/            one factory per stage (stt/llm/tts/avatar/vad)
local_services/      local avatar server + Pipecat wrapper
scripts/
  preflight.py       resolve every import (catches Pipecat version drift)
```

---

## Quick start

> **Full from-zero setup** — hardware requirements, both TTS paths (WSL2+vLLM and
> Windows-only), all conda environments, weights, and `.env`: see **[`INSTALL.md`](INSTALL.md)**.

```bash
pip install -r requirements.txt
python -m scripts.preflight       # verify imports resolve BEFORE wiring keys
copy .env.example .env            # then fill in keys (Deepgram, OpenRouter)
```

The default stack is **3 processes** (TTS server + avatar server + pipeline). See
**`STATUS.md`** (current state, source of truth), **`WORKFLOW.md`** (full run +
`.env` reference), and **`docs/PROBLEMS-AND-FIXES.md`** (every bug found + how it was
fixed — read before re-debugging the avatar/audio).

**Quickest start:** double-click **`Run VisualLLm.exe`** in the repo root — it starts the WSL TTS,
the avatar + pipeline, and the config panel, then opens the client. Press Enter in its window to stop
everything. Manual version:

```bash
# 1. CosyVoice TTS — vLLM in WSL (TTFB ~1.1s). Then set COSYVOICE_URL to the WSL IP (`wsl hostname -I`),
#    NOT localhost (WSL2's localhost relay buffers the audio stream).
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"   # :8001
# 2 + 3. MuseTalk avatar server + pipeline (one script: starts both, propagates the MuseTalk knobs from .env)
.\scripts\run.ps1
```

Open `http://localhost:7860/client/` (**trailing slash**), allow the mic, **wait for
the avatar face to appear**, then talk. The console logs a `[TTFO]` line per turn;
the disconnect log prints the median/p95 summary.

> **Remote viewing** is over Tailscale (`tailscale serve` HTTPS URL) in a native
> browser, never RDP. If the remote mic is flaky, that's WebRTC ICE candidate
> pollution — `WEBRTC_ICE_SUBNET=100.64.0.0/10` pins ICE to Tailscale (see STATUS.md).

> **Version note:** Pipecat's import paths shift between releases. If an import
> errors, check `python -c "import pipecat; print(pipecat.__version__)"` — the
> fragile imports are isolated to `pipeline/stages/*.py`, `pipeline/main.py`,
> and `pipeline/metrics.py`.

## Switching to Mandarin

Set `LANGUAGE=zh` in `.env` (and optionally an `OPENROUTER_MODEL` strong at
Chinese). Deepgram switches to `zh-TW` and CosyVoice speaks zh — no code changes.

> **Known limitation:** the Chinese voice starts ~1s later than English — CosyVoice's
> zh first-chunk TTFB is ~2.3s vs ~1.1s for en. It's fixable in TTS but the fix starves
> the shared-GPU avatar render, so it's left as-is. See `docs/PROBLEMS-AND-FIXES.md` P15.

---

## Measuring the goal

- `TtfoMeter` (in the pipeline) logs each turn's TTFO and a p95 summary.
  **Pass = p95 < 3 s.**
- Biggest tuning lever: the VAD `stop_secs` in `pipeline/stages/vad.py`.
