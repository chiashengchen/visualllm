# local_services

Local models running on the 5060 Ti, for Phase 2/3. Two layers:

**Pipecat client wrappers** (drop into the pipeline via `pipeline/stages/*.py`):

| Module | Class | Stage | Phase |
|--------|-------|-------|-------|
| `funasr_stt.py` | `FunASRSTTService` | Mandarin STT (in-process, Paraformer) | 2 |
| `cosyvoice_tts.py` | `CosyVoiceTTSService` | zh-TW streaming TTS (HTTP client) | 2/3 |
| `musetalk_video.py` | `MuseTalkVideoService` | local lip-sync avatar (websocket client) | 3 |

**Model servers** (the heavy processes the clients talk to):

| Folder | Port | Talks to | VRAM |
|--------|------|----------|------|
| `cosyvoice_server/` | 8001 | `CosyVoiceTTSService` | ~2 GB |
| `musetalk_server/`  | 8002 | `MuseTalkVideoService` | ~4–6 GB |

`musetalk_server/` defaults to the **TensorRT** render path (`MUSETALK_TRT=1`, ~1.5× faster — holds A/V
sync under shared-GPU contention; `docs/PROBLEMS-AND-FIXES.md` P16). Engines (`trt_cache/`, ~1.75 GB,
gitignored, GPU-specific) are built once with `musetalk_server/trt_build.py`; any load failure falls back
to PyTorch.

FunASR runs in-process (no server); the first call downloads its weights.

## Wiring them in (`.env`)

```
STT_PROVIDER=funasr
TTS_PROVIDER=cosyvoice_local      # start: python -m local_services.cosyvoice_server.app
AVATAR_PROVIDER=musetalk_local    # start: python -m local_services.musetalk_server.app
```

## Status / what's left

- `funasr_stt.py`, `cosyvoice_tts.py`, `musetalk_video.py` — **complete** client logic.
- `cosyvoice_server/app.py` — **near-complete**; needs CosyVoice installed +
  the `CosyVoice2-0.5B` checkpoint (see its `requirements.txt`).
- `musetalk_server/app.py` — websocket protocol + audio-windowing **complete**;
  the two model calls (`MuseTalkEngine.load` / `.render`) are marked `TODO[MuseTalk]`
  and map onto MuseTalk's `realtime_inference`. Until weights are wired it emits
  neutral gray frames so the transport + A/V-sync path can be tested end-to-end.

## VRAM reality (16 GB)

Don't run everything local at once: MuseTalk (~5) + CosyVoice2 (~2) + FunASR
(~2) + Qwen-7B-4bit (~6) ≈ 15 GB with no headroom. Keep **either the LLM or the
avatar on API** in steady state (decide in Phase 3 from measured numbers).
