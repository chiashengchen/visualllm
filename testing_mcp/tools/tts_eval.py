"""TTS evaluation tool: TTFB + round-trip CER.

Round-trip CER: synthesize text with TTS → transcribe with Deepgram STT → compare
to the original text. A clean-sounding voice produces low CER; garbled/mispronounced
audio produces high CER. Thresholds: CER < 5% = pass, > 15% = quality alert.
"""
from __future__ import annotations

import asyncio
import io
import os
import time
from typing import Any

import aiohttp


async def run_tts_eval(
    texts: list[str],
    language: str = "zh",
    tts_url: str = "http://localhost:8001",
    tts_voice: str = "weather",
) -> dict[str, Any]:
    """Evaluate TTS quality: measures TTFB, RTF, and round-trip CER for each text.

    Round-trip CER = synthesize → re-transcribe with Deepgram → character error rate vs original.
    CER < 5% = pass, > 15% = quality alert (consider retraining or changing reference audio).

    Args:
        texts: List of sentences to synthesize and evaluate.
        language: Language code matching the pipeline config (en/zh/th).
        tts_url: URL of the CosyVoice / MOSS TTS server.
        tts_voice: Voice name to pass in the TTS request.

    Returns:
        Dict with per-text results and aggregate statistics.
    """
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        return {"error": "DEEPGRAM_API_KEY not set — needed for round-trip CER"}

    lang_map = {"zh": "zh-TW", "en": "en-US", "th": "th"}
    dg_lang = lang_map.get(language, language)

    results = []
    for text in texts:
        result = await _eval_one(text, tts_url, tts_voice, deepgram_key, dg_lang)
        results.append(result)

    cers = [r["round_trip_cer"] for r in results if r.get("round_trip_cer") is not None]
    ttfbs = [r["ttfb_ms"] for r in results if r.get("ttfb_ms") is not None]

    return {
        "results": results,
        "aggregate": {
            "mean_cer": round(sum(cers) / len(cers), 4) if cers else None,
            "max_cer": round(max(cers), 4) if cers else None,
            "mean_ttfb_ms": round(sum(ttfbs) / len(ttfbs), 1) if ttfbs else None,
            "quality_verdict": _quality_verdict(cers),
        },
    }


async def _eval_one(
    text: str,
    tts_url: str,
    voice: str,
    deepgram_key: str,
    dg_lang: str,
) -> dict[str, Any]:
    audio_buf = io.BytesIO()
    ttfb_ms = None
    start = time.monotonic()

    try:
        async with aiohttp.ClientSession() as session:
            # --- TTS: stream raw PCM from the CosyVoice / MOSS server ---
            payload = {"text": text, "voice": voice}
            async with session.post(f"{tts_url}/tts/stream", json=payload) as resp:
                if resp.status != 200:
                    return {"text": text, "error": f"TTS HTTP {resp.status}"}
                async for chunk in resp.content.iter_chunked(4096):
                    if ttfb_ms is None:
                        ttfb_ms = (time.monotonic() - start) * 1000
                    audio_buf.write(chunk)
    except Exception as e:
        return {"text": text, "error": f"TTS request failed: {e}"}

    audio_bytes = audio_buf.getvalue()
    if not audio_bytes:
        return {"text": text, "error": "TTS returned empty audio"}

    duration_s = len(audio_bytes) / (2 * 24000)  # 16-bit PCM, 24 kHz mono
    rtf = (time.monotonic() - start) / max(duration_s, 0.001)

    # --- Round-trip: transcribe the raw PCM with Deepgram ---
    transcript = await _transcribe_pcm(audio_bytes, deepgram_key, dg_lang)
    cer = _cer(text, transcript) if transcript else None

    return {
        "text": text,
        "transcript": transcript,
        "ttfb_ms": round(ttfb_ms, 1) if ttfb_ms else None,
        "rtf": round(rtf, 3),
        "audio_duration_s": round(duration_s, 2),
        "round_trip_cer": round(cer, 4) if cer is not None else None,
        "verdict": "pass" if cer is not None and cer < 0.05 else ("alert" if cer and cer > 0.15 else "ok"),
    }


async def _transcribe_pcm(pcm_bytes: bytes, api_key: str, language: str) -> str | None:
    """Send raw 16-bit 24kHz mono PCM to Deepgram and return the transcript."""
    url = f"https://api.deepgram.com/v1/listen?model=nova-2&language={language}&encoding=linear16&sample_rate=24000&channels=1"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/raw"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=pcm_bytes, headers=headers) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception:
        return None


def _cer(reference: str, hypothesis: str) -> float:
    """Character error rate (edit distance / len(reference))."""
    if not reference:
        return 0.0
    r, h = list(reference), list(hypothesis or "")
    # Simple DP edit distance
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def _quality_verdict(cers: list[float]) -> str:
    if not cers:
        return "no_data"
    mean_cer = sum(cers) / len(cers)
    if mean_cer < 0.05:
        return "pass"
    if mean_cer > 0.15:
        return "alert — consider retraining or updating reference audio"
    return "ok"
