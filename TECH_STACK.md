# VisualLLm — Technology Stack

Real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
Multi-turn, streaming end-to-end. Goal: time-to-first-output **< 8 s**.

> Source of truth for current state is `STATUS.md`; this file is just the
> consolidated list of every technology in use.

## The active pipeline (one pure stack)

| Stage | Technology | Notes |
|-------|-----------|-------|
| **Orchestration** | [Pipecat](https://github.com/pipecat-ai/pipecat) `pipecat-ai` 1.3.0 | Real-time voice+video agent framework; wires every stage with streaming + barge-in |
| **Transport** | WebRTC (Pipecat WebRTC transport) | Browser ↔ server audio/video; serves the prebuilt client at `/client` |
| **VAD** | Silero VAD (local) | Voice-activity detection via `SileroVADAnalyzer` |
| **STT** | Deepgram nova-2 | `en-US` / `zh-TW` / `th` selected by `LANGUAGE` |
| **LLM** | OpenRouter | Any model via `OPENROUTER_MODEL` (default `google/gemini-2.5-flash-lite`); uses Pipecat's OpenAI-compatible `OpenAILLMService` |
| **TTS** | **CosyVoice2-0.5B** local streaming server (female zero-shot) — **runs on vLLM in WSL** (TTFB ~1.1s). `TTS_PROVIDER` fallbacks: ElevenLabs `eleven_flash_v2_5`, Deepgram Aura | `:8001`, separate repo `E:\Claude\cosyvoice-local-tts` |
| **Avatar** | **MuseTalk** (mouth-region lip-sync, female portrait) — default; or `AVATAR=ditto` (full-face TensorRT path, kept) / `none` (client-rendered 3D face) | `:8002`, `musetalk` conda env |

## TTS server — CosyVoice2 on vLLM (the 2026-06-22 latency fix)

| Component | Technology |
|-----------|-----------|
| Engine | **vLLM 0.23** runs CosyVoice2's autoregressive speech-token LLM (the ~3s bottleneck) → first-chunk latency 3.4s → ~1.1s |
| Runtime | **WSL2 Ubuntu 24.04** on the Blackwell RTX 5060 Ti; conda env `cosyvllm` (Python 3.10), **torch 2.11.0+cu130** (sm_120) |
| Server | **FastAPI** + **Uvicorn**, the same `app.py`; launched via `run_vllm_server.sh` |
| Reachability | pipeline → **WSL IP** (NOT localhost — WSL2's localhost relay buffers the audio stream) |
| Fallback | the original Windows `tts`-env PyTorch CosyVoice server (TTFB ~3.4s) — set `COSYVOICE_URL=http://localhost:8001` |

## Avatar GPU server (MuseTalk default; Ditto fallback)

| Component | Technology |
|-----------|-----------|
| Web server | **FastAPI** + **Uvicorn** (websocket streaming) |
| Inference | **PyTorch** + CUDA, on an RTX 5060 Ti (Blackwell sm_120) |
| ONNX runtime | **onnxruntime-gpu** (needs CUDA-12 DLLs on path or falls back to CPU) |
| Face/landmarks | **MediaPipe** (Ditto path); MuseTalk uses Whisper-chunk audio features |
| Misc | **NumPy** (replaces Ditto's Cython blend kernel — compiler-free) |
| Env | dedicated `musetalk` conda env (default) / `ditto` conda env (fallback), separate from the pipeline env |

Wire contract: client → server sends config/control JSON + 16 kHz mono PCM;
server → client returns RGB frame buffers (`MUSETALK_SIZE`/`DITTO_SIZE` px) at a steady fps.

## Supporting libraries

| Library | Purpose |
|---------|---------|
| **python-dotenv** | `.env`-driven config |
| **loguru** | Logging (`[TTFO]` metrics, etc.) |
| **websockets** | Pipeline ↔ Ditto server client |
| **NumPy** | Audio/frame buffers |
| **asyncio** | Async pipeline runtime |

## Platform / runtime

- **Pipeline env:** SYSTEM Python 3.11 on **Windows 11** (has pipecat — NOT a conda env)
- **Avatar env:** conda `musetalk` (default) / `ditto` (fallback), Python 3.10
- **TTS env:** **WSL2 Ubuntu** conda `cosyvllm` (vLLM, Python 3.10) — default; or Windows conda `tts` (fallback)
- **Compiler toolchain IS installed** on Windows (VS BuildTools MSVC 14.44 + Win SDK + CUDA Toolkit 12.8, driver kept 591.44); in WSL a conda `c-compiler` covers Triton JIT. The NumPy-blend + DLL-path workarounds are kept (harmless).
- **Conda / Miniconda** for the GPU envs (Windows + WSL)
- **Browser client** = Pipecat prebuilt bundle (`pipecat-ai-prebuilt`), served as-is

## Configuration knobs (`.env`)

- `LANGUAGE` (`en` | `zh` | `th`), `TTFO_TARGET_SECONDS` (default 8), `CHARACTER_MODE` (`0` | `1`)
- `AVATAR` (`musetalk` | `ditto` | `none`), `TTS_PROVIDER` (`cosyvoice` | `elevenlabs` | `deepgram`)
- `COSYVOICE_URL` (the WSL IP for the vLLM server; localhost for the Windows fallback), `COSYVOICE_VOICE`, `COSYVOICE_PACE_RATE`
- `WEBRTC_ICE_SUBNET` (`100.64.0.0/10` = pin ICE to Tailscale; fixes the remote mic), `WEBRTC_VIDEO_BITRATE_MAX`, `CLIENT_JITTER_BUFFER_MS`
- MuseTalk tuning: `MUSETALK_SYNC_MODE` (**`steady`** = synced start, default; `live` = voice never pauses), `MUSETALK_FPS`, `MUSETALK_SIZE`, `MUSETALK_LEAD_FRAMES`
- API keys: `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY`, `ELEVENLABS_API_KEY` (+ `ELEVENLABS_VOICE_ID`, only if `TTS_PROVIDER=elevenlabs`)
- Ditto tuning (when `AVATAR=ditto`): `DITTO_TRT`, `DITTO_FPS`, `DITTO_SIZE`, `DITTO_OVERLAP`, `DITTO_SYNC_WITH_AUDIO`, `DITTO_SYNC_LEAD_S`

## Removed / recoverable from git history

Earlier multi-provider sprawl, kept only in history: **Simli**, **HeyGen** avatars;
**F5-Thai** TTS; **FunASR**, **faster-whisper** STT; **Azure** zh path; the echo-guard.
(MuseTalk + CosyVoice are the current defaults, not removed; ElevenLabs/Deepgram/Ditto are live fallback switches.)
