# local_services

Local model integrations for the speech→avatar pipeline. Two layers: thin **Pipecat
client wrappers** (selected by `.env`, used from `pipeline/stages/*.py`) and the heavier
**model servers/engines** they talk to. The single shared GPU is ~16 GB (see
`../docs/gpu-memory-notes.md`).

## Pipecat client wrappers

| Module | Class | Stage | Runs |
|--------|-------|-------|------|
| `sherpa_stt.py` | `SherpaStreamingSTTService` | **STT** — local OFFLINE **streaming** (sherpa-onnx zipformer, bilingual zh-en); drives turn-taking from its own ASR endpoint | **in-process**, CPU/~0 VRAM |
| `funasr_stt.py` | `FunasrSTTService` | **STT** — local OFFLINE **segmented** (SenseVoice-Small); needs the energy-VAD to fire end-of-turn | HTTP client → `funasr_server` `:8004` |
| `cosyvoice_tts.py` | `CosyVoiceTTSService` | **TTS** — CosyVoice2 (also reused for MOSS via `MOSS_URL`) | HTTP client → `:8001` (or `:8003`) |
| `musetalk_video.py` | `MuseTalkVideoService` | **Avatar** — mouth-region lip-sync | websocket client → `:8002` |
| `avatar_memory.py` / `weather_chain_llm.py` | — | optional avatar memory / NCU weather-bot LLM | — |

## Model servers / engines

| Folder | Port | Env | Talks to | Notes |
|--------|------|-----|----------|-------|
| `cosyvoice_server/` *(vendored at repo `../tts/cosyvoice-server/`)* | 8001 | `cosyvllm` (WSL) / `tts` (Win) | `CosyVoiceTTSService` | default TTS; vLLM in WSL, TTFB ~1.1s |
| `musetalk_server/` | 8002 | `musetalk` conda | `MuseTalkVideoService` | lip-sync avatar; ~4–6 GB |
| `moss_server/` | 8003 | `moss-tts` conda | `CosyVoiceTTSService` (`MOSS_URL`) | alt TTS (`TTS_PROVIDER=moss`) |
| `funasr_server/` | 8004 | `funasr-stt` conda | `FunasrSTTService` | SenseVoice; CPU/~0 VRAM; `run.ps1` auto-starts when `STT_PROVIDER=funasr` |

`sherpa_stt.py` needs **no server** — sherpa-onnx + opencc run in the pipeline's system Python
(deps in `../requirements.txt`); the model lives under `../models/` (gitignored).

## Wiring (`.env`)

```
STT_PROVIDER=deepgram      # deepgram (cloud, default) | sherpa (local streaming) | funasr (local segmented)
TTS_PROVIDER=cosyvoice     # cosyvoice (default) | moss | elevenlabs | deepgram
# avatar is always MuseTalk; knobs MUSETALK_* (see ../WORKFLOW.md §8)
```

Full setup for each local option: **`../INSTALL.md`** (STT §6.5). Provider behavior + the
streaming-vs-segmented / VAD detail: **`../WORKFLOW.md`** and **`../CLAUDE.md`**.
