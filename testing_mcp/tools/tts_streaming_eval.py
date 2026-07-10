"""TTS streaming-input evaluation: mock the live pipeline's LLM→TTS feed.

The live pipeline never sends a whole reply to TTS at once: the LLM streams tokens,
the aggregator (local_services/first_piece_aggregator.py) cuts them into pieces —
the FIRST piece flushes early at a full-width ，；： once ≥5 CJK chars are buffered
(COSYVOICE_FIRST_PIECE_ZH), later pieces are whole sentences — and each piece is a
separate serial /tts/stream request. run_tts_eval (whole-sentence) can't see two
things this mode measures:

  - first_piece_ttfb_ms: time to first audio when the opener is a short clause
    (the live TTFA — should beat the whole-sentence TTFB)
  - seam gaps: assuming playback starts at the first piece's first chunk and runs
    continuously, gap(N→N+1) = piece N+1's first-audio arrival − piece N's playback
    end. A POSITIVE gap is an audible mid-reply pause (the live "句間停頓");
    negative = the next piece arrived while earlier audio still covers it.

Quality is still round-trip CER: all pieces' PCM concatenated → Deepgram → compare
to the full reference text (punctuation/whitespace stripped — Deepgram returns none
without smart_format, so comparing raw strings would inflate CER).

The splitter is a standalone replica of FirstClauseAggregator's zh rules (no pipecat
dependency — this runs on machines without the pipeline env).
"""
from __future__ import annotations

import asyncio
import io
import re
import os
import time
from typing import Any

import aiohttp

from testing_mcp.tools.tts_eval import _cer, _transcribe_pcm

# Mirror of first_piece_aggregator.py's zh rules (keep in sync):
# full-width clause punct only — deliberately not 、, and no char-cap fallback.
_ZH_CLAUSE_PUNCT = "，；："
_SENTENCE_PUNCT = "。！？!?"
_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")
_STRIP = re.compile(r"[^\w]", re.UNICODE)


def split_stream(text: str, zh_min_chars: int = 5) -> list[tuple[str, int]]:
    """Replay FirstClauseAggregator's zh piece-splitting over a finished char stream.

    First piece: flush at a full-width ，；： once ≥ zh_min_chars CJK chars are
    buffered (or at a sentence end, whichever comes first). Later pieces: whole
    sentences at 。！？. Trailing remainder flushes at end (the aggregator's flush()).

    Returns (piece_text, end_char_index) pairs — the index is how many chars of the
    original text the mock LLM must have emitted for the piece to be complete, which
    is what the emission-clock timing needs.
    """
    pieces: list[tuple[str, int]] = []
    buf = ""
    first_done = False
    for i, ch in enumerate(text):
        buf += ch
        stripped = buf.strip()
        if not first_done:
            if ch in _SENTENCE_PUNCT:
                pieces.append((stripped, i + 1))
                buf, first_done = "", True
            elif ch in _ZH_CLAUSE_PUNCT and len(_CJK.findall(stripped)) >= zh_min_chars:
                # keep the trailing ， — the zh normalizer uses it for clause prosody
                pieces.append((stripped, i + 1))
                buf, first_done = "", True
        elif ch in _SENTENCE_PUNCT:
            pieces.append((stripped, i + 1))
            buf = ""
    if buf.strip():
        pieces.append((buf.strip(), len(text)))
    return pieces


async def run_tts_streaming_eval(
    texts: list[str],
    tok_rate: float = 25.0,
    language: str = "zh",
    tts_url: str = "http://localhost:8001",
    tts_voice: str = "weather",
    zh_min_chars: int = 5,
) -> dict[str, Any]:
    """Evaluate TTS under the live pipeline's streaming feed (mocked LLM + piece splitting).

    Each text plays the role of one LLM reply: characters are emitted at tok_rate
    chars/s, split into pieces with the pipeline's first-clause zh rules, and each
    piece is a serial /tts/stream request. Reports per-reply first-piece TTFB, seam
    gaps between pieces, and round-trip CER over the concatenated audio.

    Args:
        texts: Reference replies to stream (each = one bot turn).
        tok_rate: Mock LLM emission speed in chars/second (gemini-flash zh ≈ 25).
        language: en/zh/th — picks the Deepgram language for the round trip.
        tts_url: CosyVoice /tts/stream base URL.
        tts_voice: Voice name passed to the TTS request.
        zh_min_chars: Min CJK chars before the first-piece comma flush (pipeline default 5).

    Returns:
        Dict with per-text piece timings/gaps/CER and aggregate stats.
    """
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        return {"error": "DEEPGRAM_API_KEY not set — needed for round-trip CER"}
    dg_lang = {"zh": "zh-TW", "en": "en-US", "th": "th"}.get(language, language)

    results = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            results.append(
                await _eval_streaming_one(
                    session, text, tok_rate, tts_url, tts_voice, deepgram_key, dg_lang, zh_min_chars
                )
            )

    ttfbs = [r["first_piece_ttfb_ms"] for r in results if r.get("first_piece_ttfb_ms")]
    gaps = [g for r in results for g in r.get("gaps_s", [])]
    cers = [r["round_trip_cer"] for r in results if r.get("round_trip_cer") is not None]
    return {
        "results": results,
        "aggregate": {
            "mean_first_piece_ttfb_ms": round(sum(ttfbs) / len(ttfbs), 1) if ttfbs else None,
            "worst_gap_s": round(max(gaps), 3) if gaps else None,
            "positive_gaps": sum(1 for g in gaps if g > 0.05),
            "mean_cer": round(sum(cers) / len(cers), 4) if cers else None,
            "tok_rate_chars_per_s": tok_rate,
        },
    }


async def _eval_streaming_one(
    session: aiohttp.ClientSession,
    text: str,
    tok_rate: float,
    tts_url: str,
    voice: str,
    deepgram_key: str,
    dg_lang: str,
    zh_min_chars: int,
) -> dict[str, Any]:
    # --- Mock LLM: emit chars at tok_rate; a piece becomes ready when the splitter
    # would flush it, i.e. after its last char has "arrived". We precompute the split,
    # then timestamp each piece's readiness from the emission clock — equivalent to
    # incremental splitting, without duplicating the aggregator's char loop.
    split = split_stream(text, zh_min_chars)
    if not split:
        return {"text": text, "error": "splitter produced no pieces"}
    char_interval = 1.0 / max(tok_rate, 0.1)
    pieces = [p for p, _ in split]
    ready_at = [end_idx * char_interval for _, end_idx in split]

    t0 = time.monotonic()
    piece_rows = []
    all_pcm = io.BytesIO()
    playback_end = None  # absolute time the continuous playback runs out of audio
    gaps: list[float] = []

    for idx, piece in enumerate(pieces):
        # wait until the mock LLM has finished emitting this piece
        wait = ready_at[idx] - (time.monotonic() - t0)
        if wait > 0:
            await asyncio.sleep(wait)

        req_start = time.monotonic()
        ttfb = None
        buf = io.BytesIO()
        async with session.post(f"{tts_url}/tts/stream", json={"text": piece, "voice": voice}) as resp:
            if resp.status != 200:
                return {"text": text, "error": f"TTS HTTP {resp.status} on piece {idx}: {piece!r}"}
            async for chunk in resp.content.iter_chunked(4096):
                if ttfb is None:
                    ttfb = time.monotonic() - req_start
                buf.write(chunk)

        pcm = buf.getvalue()
        audio_s = len(pcm) / (2 * 24000)
        first_audio_at = req_start + (ttfb or 0)

        if playback_end is None:
            playback_end = first_audio_at + audio_s  # playback starts at first piece's first chunk
        else:
            gaps.append(round(first_audio_at - playback_end, 3))
            playback_end = max(playback_end, first_audio_at) + audio_s

        all_pcm.write(pcm)
        piece_rows.append({
            "piece": piece,
            "ttfb_ms": round((ttfb or 0) * 1000, 1),
            "audio_s": round(audio_s, 2),
        })

    audio_bytes = all_pcm.getvalue()
    transcript = await _transcribe_pcm(audio_bytes, deepgram_key, dg_lang)
    ref = _STRIP.sub("", text)
    hyp = _STRIP.sub("", transcript or "")
    cer = _cer(ref, hyp) if transcript else None
    total_audio_s = len(audio_bytes) / (2 * 24000)
    spoken_chars = len(_STRIP.sub("", text))

    return {
        "text": text,
        "transcript": transcript,
        "pieces": piece_rows,
        "first_piece_ttfb_ms": piece_rows[0]["ttfb_ms"],
        "gaps_s": gaps,
        "max_gap_s": max(gaps) if gaps else None,
        "round_trip_cer": round(cer, 4) if cer is not None else None,
        "dur_ratio_s_per_char": round(total_audio_s / max(spoken_chars, 1), 3),
    }
