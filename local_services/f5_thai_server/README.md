# F5-TTS-THAI server — the self-host TTS spike

**Purpose:** de-risk the question *"can we build our own Thai emotional voice model?"* by standing up
the best open Thai TTS (`VIZINTZOR/F5-TTS-THAI`) locally and A/B-ing it against ElevenLabs — **before**
committing weeks to building/fine-tuning one. See the verdict in `../../../visualllm-business/` docs.

This is a **spike, not production.** It runs in its **own isolated venv** so its heavy ML deps can't
break the parent pipeline's working global `torch`/`pipecat`.

## Why isolated venv (and on C:)

- The parent repo runs on **global** packages. Installing `f5-tts` globally risks pulling an
  incompatible `torch`/`numpy` and breaking the live pipeline. So: dedicated venv.
- Heavy ML deps need ~10–12 GB. The venv lives at **`E:/f5-spike/.venv-f5`** (E: was freed up for it).
  A venv isn't natively relocatable, so always invoke it by absolute path via
  `E:/f5-spike/.venv-f5/Scripts/python.exe -m pip ...` (the `pip.exe` shim hardcodes its build path).

## Setup (done once on this box; venv already at `E:/f5-spike/.venv-f5`)

```bash
py -3.11 -m venv E:/f5-spike/.venv-f5
# Blackwell / RTX 5060 Ti needs the CUDA 12.8 wheels:
E:/f5-spike/.venv-f5/Scripts/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
E:/f5-spike/.venv-f5/Scripts/python.exe -m pip install -r requirements.txt
winget install --id Gyan.FFmpeg -e        # pydub needs the ffmpeg CLI on PATH
```

**Status: verified working.** Loads the 1M-step checkpoint on the 5060 Ti and synthesizes Thai at
**RTF ≈ 0.70** (faster than real time). `ref_text.txt` (the reference clip's transcript,
pre-computed with faster-whisper) is shipped so inference never touches the torchcodec-based Whisper.

## Run the two things

**1) The A/B ears test (fastest answer — do this first):**
```bash
E:/f5-spike/.venv-f5/Scripts/python.exe -m local_services.f5_thai_server.ab_thai_voice
# writes f5th_*.wav into visualllm-business/voice-samples/ — compare vs Bella.mp3 by ear
```

**2) The live server (to drive the pipeline locally):**
```bash
E:/f5-spike/.venv-f5/Scripts/python.exe -m local_services.f5_thai_server.server
# smoke test:
curl -X POST http://localhost:8001/tts -H "Content-Type: application/json" \
     -d '{"text":"สวัสดีค่ะ ยินดีที่ได้รู้จักนะคะ"}' --output out.wav
```
Then point the pipeline at it: set `TTS_PROVIDER=f5_thai_local` (uses `F5_THAI_URL`, default
`http://localhost:8001`) and run `python -m pipeline.main`. The `f5_thai_local` branch in
`pipeline/stages/tts.py` reuses the model-agnostic `CosyVoiceTTSService` HTTP client.

## Auditioning the *character* voice (voice cloning)

F5-TTS is zero-shot: the **reference clip = the voice**. Drop a short (~10–15 s), **license-clear**
Thai `.wav` and its transcript via env to clone that voice:
```bash
F5_THAI_REF_AUDIO=C:/path/to/ref.wav  F5_THAI_REF_TEXT="ข้อความในคลิปอ้างอิง"
```
Leaving them blank uses the model's bundled reference.

## Knobs (env)

| var | default | meaning |
|---|---|---|
| `F5_THAI_REPO` | `VIZINTZOR/F5-TTS-THAI` | HF repo |
| `F5_THAI_CKPT` | `model_500000.pt` | checkpoint file in the repo (check the repo for the latest) |
| `F5_THAI_VOCAB` | `vocab.txt` | vocab file |
| `F5_THAI_REF_AUDIO` / `F5_THAI_REF_TEXT` | bundled | reference clip = cloned voice |
| `F5_THAI_PORT` / `F5_THAI_URL` | `8001` | server port / pipeline target |

## Gotchas (verified on this box — RTX 5060 Ti / Blackwell, torch 2.11+cu128)

- **Import order: `import torch` BEFORE `f5_tts`.** `f5_tts.api` imports `soundfile` first; with the
  bleeding-edge cu128 stack the native load order otherwise corrupts memory → **hard segfault
  (exit 139)** with no Python traceback. Both `server.py` and `ab_thai_voice.py` import torch first as
  a guard — keep it. (Driving inference via `f5_tts.infer.utils_infer` directly also avoids it.)
- **`numpy<2`.** Pinned to 1.26.4 — several native deps in this stack aren't NumPy-2 ABI-clean.
- **Checkpoint name:** the latest is top-level **`model_1000000.pt`** (1M steps). The repo also nests
  copies under `model/` and `old_small_model/`; if a name 404s, `list_repo_files(...)` and update.
- **Arch:** load with `model="F5TTS_v1_Base"`. A wrong arch yields a clear state-dict shape error
  (not a crash) — fall back to `F5TTS_Base` if so.
- **ffmpeg warning** from pydub is harmless for wav refs (soundfile handles wav directly).

## Known limits (read before judging)

- **Emotion < ElevenLabs v3.** F5-TTS carries emotion through the *reference clip's* prosody, not
  `[tags]`. A genuinely expressive Thai voice may need fine-tuning on collected Thai emotional speech.
- **Not natively streaming** → higher TTFO than CosyVoice2. If it can't clear the ~3s "feels live"
  bar for live chat, it's still fine for the **non-live** showcase voice. Measure, don't assume.
- **Checkpoint name drifts** — if `model_500000.pt` 404s, list the repo's files and update `F5_THAI_CKPT`.
