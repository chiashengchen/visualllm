# Local Offline STT (SenseVoice on CPU) — Design

**Date:** 2026-06-30
**Branch:** `feat/offline-stt-sensevoice` (off `main` — avatar-system work, destined for public main)
**Goal:** Replace cloud Deepgram STT with a **fully-offline, local** Mandarin (zh-TW) STT that adds
**~0 GPU VRAM**, keeping Deepgram as a one-line fallback. Plus low-risk VRAM-tightening on the
already-full 16GB card.

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Approach | **B** — a local FunASR/SenseVoice server + a custom Pipecat STT wrapper (mirrors the CosyVoice/MOSS pattern), not the built-in Whisper. |
| Model | **SenseVoice-Small** (`iic/SenseVoiceSmall`, FunAudioLLM family, same as CosyVoice) — ~234M non-autoregressive, fast, strong Mandarin. |
| Placement | **CPU (system RAM)** — `device="cpu"`, ~0 VRAM. STT runs before the avatar render, so CPU work doesn't collide with it. |
| zh-TW output | SenseVoice emits Simplified; convert to Traditional with **OpenCC `s2twp`** in the STT wrapper. |
| Default behavior | `STT_PROVIDER=deepgram` stays the default; offline STT is **opt-in** (`STT_PROVIDER=funasr`). Public default unchanged. |
| VRAM work | Included as a separate low-risk task: investigate the 254-MiB-free reading, add `expandable_segments`, document the vLLM KV-trim knobs. |

## Why CPU/RAM (the key constraint)

`nvidia-smi` shows the card **full now: 15797 / 16311 MiB used, 254 MiB free** (vLLM CosyVoice +
MuseTalk). There is no headroom for a GPU STT model. The existing TTS/avatar **cannot** move to RAM
(real-time GPU inference needs weights in VRAM; CPU-offload pages over PCIe and destroys latency). But
**STT can**: SenseVoice-Small on CPU uses system RAM, ~0 VRAM, and — because STT runs while/just after
the user speaks, *before* the LLM→TTS→avatar render — its CPU cost does not run concurrently with the
avatar (unlike the prior CPU-LLM contention episode). Honest cost: segmented (no interim partials),
+~0.3–1.5s to first transcript on CPU, zh-TW via OpenCC.

## Architecture

Each unit is isolated and mirrors the existing single-provider-factory + local-server patterns.

### 1. `local_services/funasr_server/app.py` — the STT server
- FastAPI server, port **:8004**, in its own **`funasr-stt` conda env** (heavy ML deps isolated from
  the system pipeline env, exactly like CosyVoice/MOSS).
- Loads SenseVoice-Small once via FunASR `AutoModel(model=FUNASR_MODEL, device="cpu")`.
- Endpoint **`POST /stt`**: body = raw 16 kHz mono PCM (one buffered utterance) → JSON `{"text": "..."}`.
  (Segmented, not streaming — matches SenseVoice's offline nature; the per-utterance call is simple
  and robust, no WS needed.)
- `GET /health` for the launcher to wait on (same contract as the CosyVoice/MOSS health checks).
- Reads `FUNASR_MODEL` / `FUNASR_DEVICE` from the OS env only (no python-dotenv), like the other servers.

### 2. `pipeline/stages/funasr_stt.py` — the Pipecat STT service
- `class FunasrSTTService(SegmentedSTTService)` — Pipecat buffers the utterance between VAD
  start/stop; on stop, `run_stt(audio: bytes)` POSTs the PCM to `FUNASR_URL/stt`, gets back the
  **already-Traditional** text (the server runs OpenCC — see §5), and yields a
  `TranscriptionFrame(text, user_id, timestamp)` — the exact frame the downstream aggregator/LLM
  already consume. No other pipeline change, and the pipeline env gains no OpenCC dep.
- Empty transcript → yield nothing (no crash). Network/`:8004`-down → log + yield nothing
  (degrade gracefully, never crash the turn).

### 3. `pipeline/stages/stt.py` — provider switch
- `build_stt(cfg)` branches on `cfg.stt_provider`: `"deepgram"` (default, the current code, unchanged)
  | `"funasr"` → `FunasrSTTService(url=cfg.funasr_url, ...)`. Same thin-factory style as `tts.py`.

### 4. `pipeline/config.py` — new knobs
- `stt_provider: str = _get("STT_PROVIDER", "deepgram")`
- `funasr_url: str = _get("FUNASR_URL", "http://localhost:8004")`
- `funasr_model: str = _get("FUNASR_MODEL", "iic/SenseVoiceSmall")`
- `funasr_device: str = _get("FUNASR_DEVICE", "cpu")`
- `.env.example` documents all four with the offline/zh-TW notes.

### 5. Dependencies
- **Server env (`funasr-stt`):** `funasr`, `torch` (CPU build is fine), `opencc` (or
  `opencc-python-reimplemented`), `fastapi`, `uvicorn`, `onnxruntime` (SenseVoice ONNX path).
  A `local_services/funasr_server/requirements.txt` documents these (not added to the root pipeline
  `requirements.txt`, which stays light).
- **Pipeline env:** `opencc` is needed there too IF conversion happens client-side. **Decision:** do the
  OpenCC conversion **in the server** (it already has the heavy env) and return Traditional text, so the
  pipeline env gains **no** new dep. The `funasr_stt.py` wrapper then just relays text. (Revises §2: the
  OpenCC step lives in the server, not the wrapper.)

### 6. Run / docs
- `scripts/run.ps1` and the launcher gain an optional step to start `:8004` when `STT_PROVIDER=funasr`.
- `INSTALL.md` / `WORKFLOW.md` document the offline-STT path and the `funasr-stt` env setup.

## VRAM-tightening (separate task, low-risk)
1. **Investigate the 254-MiB-free reading** — `nvidia-smi` (or Task Manager) to see whether a stale
   process / browser / prior run holds VRAM that can be reclaimed for free. Document findings.
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — set for the avatar/pipeline processes to cut
   allocator fragmentation (lets the same workload fit in less reserved VRAM). Propagate via `run.ps1`.
3. **Document the vLLM KV-trim knobs** (`COSYVOICE_VLLM_GPU_UTIL`, `--max-model-len`) in CLAUDE.md/
   WORKFLOW — these can claw back ~0.5–1.5 GB but 0.2 already crashed, so document the safe floor, don't
   change the default blindly.
   These are documentation + one env flag — no behavior change to the tuned TTS/avatar.

## Out of scope
- GPU placement of STT (rejected — card is full; CPU is the design).
- CPU-offloading the TTS/avatar weights (not viable for real-time).
- Streaming/interim-partial STT (SenseVoice is segmented; FunASR Paraformer streaming is a later option
  if interim partials become necessary).
- Thai/English tuning (SenseVoice handles en/zh multilingually; zh-TW is the target this spec optimizes).

## Risks / notes
- **CPU transcribe latency** adds to the <8s TTFO budget (~0.3–1.5s). Acceptable; measure with the
  `scripts.measure` harness and note the real number.
- **SenseVoice model download** (~1GB) needs ModelScope/HF; the conda cert-store gotcha
  (`SSL_CERT_FILE`=certifi) may apply, like the MuseTalk/CosyVoice weights.
- **First-call warmup** — load the model at server startup (lifespan), not on first request, so the
  first turn isn't penalized.
