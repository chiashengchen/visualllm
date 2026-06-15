"""F5-TTS-THAI inference server (the self-host SPIKE).

Stands up the open Thai TTS model `VIZINTZOR/F5-TTS-THAI` behind an HTTP `/tts`
endpoint whose request/response contract MIRRORS the one CosyVoiceTTSService
already speaks (see ../cosyvoice_tts.py): JSON {text, voice, sample_rate} -> audio.

That means the existing Pipecat client can drive this with near-zero new code:
just point COSYVOICE_URL (or the new f5_thai_local branch) at this server.

This is a de-risking spike, not production: it answers "is an open Thai model
good enough to be our own voice yet?" before committing to building one.

Run (in an ISOLATED venv so it can't break the parent repo's global packages):
    E:/f5-spike/.venv-f5/Scripts/python.exe -m local_services.f5_thai_server.server
Then smoke-test:
    curl -X POST http://localhost:8001/tts -H "Content-Type: application/json" \
         -d '{"text":"สวัสดีค่ะ ยินดีที่ได้รู้จักนะคะ"}' --output out.wav
"""
from __future__ import annotations

# MUST be first: torch-before-f5_tts segfault guard + torchcodec bypass + ffmpeg PATH. See _compat.
from . import _compat  # noqa: F401

import io
import os
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

# --- model identity (override via env) ---------------------------------------
HF_REPO = os.getenv("F5_THAI_REPO", "VIZINTZOR/F5-TTS-THAI")
CKPT_NAME = os.getenv("F5_THAI_CKPT", "model_1000000.pt")   # newest top-level checkpoint (1M steps)
VOCAB_NAME = os.getenv("F5_THAI_VOCAB", "vocab.txt")
MODEL_ARCH = os.getenv("F5_THAI_ARCH", "F5TTS_v1_Base")     # arch the checkpoint was trained with
# A short Thai reference clip + its transcript define the *target voice* (zero-shot clone).
# Blank -> use the repo's bundled sample/ref_audio.wav. Drop your own license-clear clip to
# audition the actual character voice.
REF_AUDIO = os.getenv("F5_THAI_REF_AUDIO", "")
# Transcript of the bundled reference clip. Pre-computed (faster-whisper) into ref_text.txt so we
# never invoke the torchcodec-based Whisper path at inference time. Override via F5_THAI_REF_TEXT.
_REF_TXT_FILE = Path(__file__).with_name("ref_text.txt")
REF_TEXT = os.getenv("F5_THAI_REF_TEXT") or (
    _REF_TXT_FILE.read_text(encoding="utf-8").strip() if _REF_TXT_FILE.exists() else ""
)
PORT = int(os.getenv("F5_THAI_PORT", "8001"))

app = FastAPI(title="F5-TTS-THAI spike")
_model = None       # lazy singleton
_ref_audio = None   # resolved reference clip path


def _load_model():
    """Load F5-TTS with the Thai finetune checkpoint pulled from the HF hub."""
    global _model, _ref_audio
    if _model is not None:
        return _model
    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    ckpt = hf_hub_download(HF_REPO, CKPT_NAME)
    vocab = hf_hub_download(HF_REPO, VOCAB_NAME)
    _ref_audio = REF_AUDIO or hf_hub_download(HF_REPO, "sample/ref_audio.wav")
    _model = F5TTS(model=MODEL_ARCH, ckpt_file=ckpt, vocab_file=vocab)
    return _model


def _pcm_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Float32 [-1,1] -> 16-bit PCM WAV bytes (what the spike returns)."""
    pcm = (np.clip(np.asarray(wav, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TtsRequest(BaseModel):
    text: str
    voice: str = "default"        # accepted for contract-compatibility (ref clip = the voice here)
    sample_rate: int = 24000


@app.get("/health")
def health():
    return {"ok": True, "model": HF_REPO, "ckpt": CKPT_NAME, "loaded": _model is not None}


@app.post("/tts")
def tts(req: TtsRequest):
    model = _load_model()
    wav, sr, _ = model.infer(
        ref_file=_ref_audio,
        ref_text=REF_TEXT,
        gen_text=req.text,
        remove_silence=True,
    )
    return Response(content=_pcm_wav_bytes(wav, sr), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    _load_model()   # warm at boot so the first /tts isn't a cold-load surprise
    uvicorn.run(app, host="0.0.0.0", port=PORT)
