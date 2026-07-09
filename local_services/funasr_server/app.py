"""Local OFFLINE STT server: SenseVoice-Small via FunASR, on CPU (~0 VRAM).

Mirrors the CosyVoice/MOSS local-server pattern. Serves :8004. The pipeline reaches it
via FUNASR_URL when STT_PROVIDER=funasr. Returns Traditional (zh-TW) text (OpenCC s2twp)
so the pipeline needs no conversion. Reads FUNASR_MODEL / FUNASR_DEVICE from the OS env
ONLY (no python-dotenv), like the other servers.

Run (in the `funasr-stt` conda env):
    python -m uvicorn local_services.funasr_server.app:app --host 0.0.0.0 --port 8004
If model download hits the conda cert store, set SSL_CERT_FILE to certifi's cacert.pem.
"""
from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request

# SenseVoice's rich postprocess injects emotion/event EMOJIS (e.g. happy/sad/applause).
# Those are not spoken words -- feeding them to the LLM as "user text" is noise, so strip
# all emoji/pictograph codepoints, keeping only the transcript (CJK + ASCII + punctuation).
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0000FE00-\U0000FE0F\U00002B00-\U00002BFF]"
)


def _strip_emoji(text: str) -> str:
    return _EMOJI.sub("", text).strip()

MODEL_ID = os.environ.get("FUNASR_MODEL", "iic/SenseVoiceSmall")
DEVICE = os.environ.get("FUNASR_DEVICE", "cpu")

_state: dict = {}


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """16 kHz mono int16 PCM bytes -> float32 [-1, 1] mono, the form FunASR expects."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


@asynccontextmanager
async def lifespan(app: FastAPI):
    from funasr import AutoModel
    from opencc import OpenCC

    # Warm the model at startup so the first turn isn't penalized.
    _state["model"] = AutoModel(model=MODEL_ID, device=DEVICE, disable_update=True)
    _state["s2tw"] = OpenCC("s2twp")  # Simplified -> Traditional (Taiwan, with phrases)
    print(f"[funasr] SenseVoice ready: {MODEL_ID} on {DEVICE}", flush=True)
    yield
    _state.clear()


app = FastAPI(title="Local SenseVoice STT", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if "model" in _state else "loading", "model": MODEL_ID, "device": DEVICE}


@app.post("/stt")
async def stt(request: Request):
    pcm = await request.body()
    if not pcm:
        return {"text": ""}
    audio = _pcm16_to_float32(pcm)
    # SenseVoice: language="auto" detects zh; use_itn adds punctuation/inverse-text-norm.
    res = _state["model"].generate(input=audio, cache={}, language="auto", use_itn=True)
    raw = res[0]["text"] if res else ""
    # SenseVoice prefixes rich tags like <|zh|><|NEUTRAL|>...; strip them, then s2twp.
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    clean = _strip_emoji(rich_transcription_postprocess(raw))
    return {"text": _state["s2tw"].convert(clean)}
