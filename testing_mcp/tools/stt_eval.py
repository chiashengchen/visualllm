"""STT evaluation tool: WER and CER against reference transcripts.

Accepts a list of (audio_file_path_or_url, reference_transcript) pairs,
transcribes each with Deepgram, and returns WER / CER.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiohttp


async def run_stt_eval(
    audio_files: list[str],
    transcripts: list[str],
    language: str = "zh",
) -> dict[str, Any]:
    """Evaluate STT accuracy against reference transcripts.

    Transcribes each audio file with Deepgram nova-2 and computes WER (word error rate)
    and CER (character error rate) against the provided reference transcripts.

    Args:
        audio_files: List of local file paths or HTTP URLs to audio files (WAV/MP3/etc).
        transcripts: List of reference transcripts (same length as audio_files).
        language: Language code (en/zh/th).

    Returns:
        Dict with per-file results and aggregate WER/CER.
    """
    if len(audio_files) != len(transcripts):
        return {"error": "audio_files and transcripts must have the same length"}

    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        return {"error": "DEEPGRAM_API_KEY not set"}

    lang_map = {"zh": "zh-TW", "en": "en-US", "th": "th"}
    dg_lang = lang_map.get(language, language)

    results = []
    for audio_path, ref in zip(audio_files, transcripts):
        result = await _eval_file(audio_path, ref, deepgram_key, dg_lang)
        results.append(result)

    wers = [r["wer"] for r in results if r.get("wer") is not None]
    cers = [r["cer"] for r in results if r.get("cer") is not None]

    return {
        "results": results,
        "aggregate": {
            "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "mean_cer": round(sum(cers) / len(cers), 4) if cers else None,
            "verdict": "pass" if wers and sum(wers) / len(wers) < 0.1 else "needs_improvement",
        },
    }


async def _eval_file(
    audio_path: str,
    reference: str,
    api_key: str,
    language: str,
) -> dict[str, Any]:
    try:
        audio_bytes, content_type = await _load_audio(audio_path)
    except Exception as e:
        return {"file": audio_path, "error": f"Could not load audio: {e}"}

    url = f"https://api.deepgram.com/v1/listen?model=nova-2&language={language}&punctuate=true"
    headers = {"Authorization": f"Token {api_key}", "Content-Type": content_type}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=audio_bytes, headers=headers) as resp:
                if resp.status != 200:
                    return {"file": audio_path, "error": f"Deepgram HTTP {resp.status}"}
                data = await resp.json()
                hypothesis = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        return {"file": audio_path, "error": f"Deepgram request failed: {e}"}

    wer = _wer(reference, hypothesis)
    cer = _cer(reference, hypothesis)

    return {
        "file": audio_path,
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
    }


async def _load_audio(path: str) -> tuple[bytes, str]:
    if path.startswith("http://") or path.startswith("https://"):
        async with aiohttp.ClientSession() as session:
            async with session.get(path) as resp:
                resp.raise_for_status()
                return await resp.read(), resp.content_type or "audio/wav"
    else:
        data = Path(path).read_bytes()
        suffix = Path(path).suffix.lower()
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(suffix.lstrip("."), "audio/wav")
        return data, mime


def _wer(reference: str, hypothesis: str) -> float:
    r = reference.split()
    h = hypothesis.split()
    if not r:
        return 0.0
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


def _cer(reference: str, hypothesis: str) -> float:
    r, h = list(reference), list(hypothesis or "")
    if not r:
        return 0.0
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
