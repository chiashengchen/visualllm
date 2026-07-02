"""End-to-end pipeline latency tool.

Drives a single turn through the pipeline by sending a pre-recorded audio file
and measuring time-to-first-audio-output (TTFO). Uses the pipeline's existing
WebRTC endpoint — so the pipeline must be running.

NOTE: True TTFO measurement requires WebRTC signaling (ICE, DTLS, etc.) which
is non-trivial from Python. This tool uses a simpler HTTP probe approach:
POST the audio to a /probe endpoint that short-circuits the WebRTC path and
returns timing breakdowns. If /probe is not available, it falls back to reporting
that it can't measure without WebRTC.
"""
from __future__ import annotations

import time
from typing import Any

import aiohttp


async def run_e2e_latency(
    audio_input: str,
    pipeline_url: str = "http://localhost:7860",
) -> dict[str, Any]:
    """Measure end-to-end pipeline latency for a single turn.

    Sends a WAV file to the pipeline's /probe endpoint and returns timing breakdowns
    for STT, LLM, and TTS stages. The pipeline must be running and ENABLE_AVATAR=0.

    Args:
        audio_input: Local path to a WAV file containing the user's speech.
        pipeline_url: Base URL of the running pipeline (default: http://localhost:7860).

    Returns:
        Dict with stage timings and total TTFO in milliseconds.
    """
    from pathlib import Path

    probe_url = f"{pipeline_url}/probe/latency"

    try:
        audio_bytes = Path(audio_input).read_bytes()
    except Exception as e:
        return {"error": f"Could not read audio file {audio_input!r}: {e}"}

    start = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("audio", audio_bytes, filename="input.wav", content_type="audio/wav")
            async with session.post(probe_url, data=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 404:
                    return {
                        "error": (
                            "Pipeline /probe/latency endpoint not found. "
                            "The probe endpoint is not yet implemented in pipeline/main.py — "
                            "use scripts/measure.py for latency measurement instead."
                        ),
                        "suggestion": "Run: python -m scripts.measure --offline-capture",
                    }
                if resp.status != 200:
                    return {"error": f"Pipeline HTTP {resp.status}"}
                result = await resp.json()
                result["wall_ms"] = round((time.monotonic() - start) * 1000, 1)
                return result
    except aiohttp.ClientConnectorError:
        return {
            "error": f"Could not connect to pipeline at {pipeline_url}. Is it running?",
            "suggestion": "Start with: python -m pipeline.main",
        }
    except Exception as e:
        return {"error": f"Probe request failed: {e}"}
