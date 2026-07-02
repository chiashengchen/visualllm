"""CosyVoice2 streaming TTS as a Pipecat service.

This is a thin HTTP client: it streams text to a local CosyVoice2 server (the
user's cosyvoice-local-tts FastAPI server, /tts/stream endpoint) and yields audio
chunks as soon as they arrive, so the avatar can start lip-syncing on the first
chunk -- the streaming path that keeps the <8s time-to-first-output budget.

The server returns raw 16-bit PCM mono at `sample_rate` (default 24 kHz, which is
CosyVoice2's native rate). Pipecat resamples downstream (to 16 kHz for the avatar).
The default voice "weather" is the server's registered female Mandarin zero-shot
reference -- CosyVoice2-0.5B is zero-shot only (no SFT preset speakers).
"""
from __future__ import annotations

from typing import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.tts_service import TTSService


class CosyVoiceTTSService(TTSService):
    def __init__(
        self,
        *,
        base_url: str,
        voice: str = "weather",          # the server's registered female zero-shot speaker
        sample_rate: int = 24000,
        **kwargs,
    ):
        # push_start/stop_frames=True so pipecat emits exactly one TTSStartedFrame +
        # TTSStoppedFrame per bot TURN (not per sentence). The avatar (musetalk_video.py)
        # keys its per-turn reset + the server's speech_start/speech_end on these; without
        # them a long multi-sentence reply never resets and drifts out of sync.
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._session: aiohttp.ClientSession | None = None

        # TTFO knob: emit a short opening clause first so the bot starts speaking ~0.8s
        # sooner (CosyVoice's first-chunk latency scales with input sentence length).
        # OFF by default; see local_services/first_piece_aggregator.py for the why + tuning.
        import os
        if os.getenv("COSYVOICE_FIRST_PIECE", "0").lower() in ("1", "true", "yes", "on"):
            from local_services.first_piece_aggregator import FirstClauseAggregator

            self._text_aggregator = FirstClauseAggregator(
                min_chars=int(os.getenv("COSYVOICE_FIRST_PIECE_MIN_CHARS", "24") or "24"),
                max_chars=int(os.getenv("COSYVOICE_FIRST_PIECE_MAX_CHARS", "60") or "60"),
                aggregation_type=self._text_aggregator.aggregation_type,
            )
            logger.info(
                "CosyVoice first-clause early-flush ON "
                f"(min={self._text_aggregator._min_chars}, max={self._text_aggregator._max_chars})"
            )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stop(self, frame):  # close the session on pipeline shutdown
        await super().stop(frame)
        if self._session and not self._session.closed:
            await self._session.close()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        # pipecat 1.3.0 calls run_tts(text, context_id) and (push_start_frame
        # default) the base class yields TTSStarted/Stopped + manages the audio
        # context -- so we ONLY yield audio frames tagged with context_id (mirrors
        # DeepgramHttpTTSService). Yielding our own start/stop frames would double them.
        logger.debug(f"CosyVoice TTS [{text}]")
        try:
            await self.start_ttfb_metrics()
            session = await self._get_session()
            payload = {
                "text": text,
                "voice": self._voice,
                "sample_rate": self.sample_rate,
            }
            async with session.post(f"{self._base_url}/tts/stream", json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    yield ErrorFrame(f"CosyVoice server {resp.status}: {body}")
                    return

                await self.start_tts_usage_metrics(text)
                first = True
                # Server streams raw PCM; read fixed-size chunks (~20ms frames).
                chunk_bytes = int(self.sample_rate * 2 * 0.02)
                async for chunk in resp.content.iter_chunked(chunk_bytes):
                    if not chunk:
                        continue
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
        except Exception as e:  # noqa: BLE001
            logger.exception("CosyVoice TTS failed")
            yield ErrorFrame(f"CosyVoice TTS error: {e}")
