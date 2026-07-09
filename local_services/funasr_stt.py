"""Pipecat STT wrapper for the local SenseVoice server (funasr_server/app.py).

A SegmentedSTTService: Pipecat buffers the utterance between VAD start/stop, then calls
run_stt(audio) once. We POST the raw PCM to FUNASR_URL/stt and emit the Traditional
(zh-TW) text it returns. Mirrors local_services/cosyvoice_tts.py (the local-server client
pattern). Degrades gracefully: a down server or empty transcript yields nothing -- never
crashes the turn.
"""
from __future__ import annotations

from typing import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class FunasrSTTService(SegmentedSTTService):
    def __init__(self, *, base_url: str, **kwargs):
        # The server picks model + auto-detects language; we declare them so Pipecat's
        # STTSettings.validate_complete doesn't log a (harmless) NOT_GIVEN error.
        kwargs.setdefault(
            "settings", STTSettings(model="SenseVoiceSmall", language=None, extra={})
        )
        super().__init__(**kwargs)
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def stop(self, frame):
        if self._session and not self._session.closed:
            await self._session.close()
        await super().stop(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None  # SegmentedSTTService contract: first yield is a None heartbeat
        try:
            session = await self._get_session()
            async with session.post(f"{self._base_url}/stt", data=audio) as resp:
                resp.raise_for_status()
                text = (await resp.json()).get("text", "").strip()
        except Exception as e:  # server down / network -> degrade, don't crash the turn
            logger.warning(f"FunASR STT call failed: {e}")
            return
        if text:
            yield TranscriptionFrame(text, "", time_now_iso8601())
