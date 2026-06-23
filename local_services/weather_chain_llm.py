"""LLM stage variant: a dedicated Chinese weather bot backed by a remote LangServe
weather-chain endpoint. It drops into the same pipeline slot as the OpenRouter LLM --
consumes LLMContextFrame, emits LLMFullResponseStart/Text/End -- so TTS, the avatar,
and the assistant aggregator are unchanged.

The chain accepts ONLY {"query","model"} (no history), so the virtual human's memory
lives in the optional MemoryStore wrapped around this service (see avatar_memory.py).
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


def extract_sse_text(data: str) -> Optional[str]:
    """Pull the text out of one LangServe SSE `data:` payload, tolerantly.

    LangServe /stream emits `event: data` + `data: <json>`. The json may be a bare
    string ("明天") or an object ({"content": ...} / {"output": ...}). Returns the text
    piece, or None for control payloads ([DONE], empty, metadata, unparseable-non-text).
    """
    s = data.strip()
    if not s or s == "[DONE]":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return s or None  # some chains stream raw text after `data: `
    if isinstance(obj, str):
        return obj or None
    if isinstance(obj, dict):
        for key in ("content", "output", "text", "answer"):
            v = obj.get(key)
            if isinstance(v, str) and v:
                return v
    return None


class WeatherChainLLMService(LLMService):
    """Streams answers from the remote LangServe weather chain, optionally rewriting
    the user's utterance through a local MemoryStore first (context engineering)."""

    def __init__(self, *, url: str, model: str, memory=None, **kwargs):
        super().__init__(**kwargs)
        self._url = url.rstrip("/") + "/stream"
        self._model = model
        self._memory = memory
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    @staticmethod
    def _last_user_text(context) -> str:
        for msg in reversed(context.get_messages()):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role != "user":
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return ""
        return ""

    async def _stream_chain(self, query: str) -> AsyncIterator[str]:
        payload = {"input": {"query": query, "model": self._model}}
        async with self._client.stream("POST", self._url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                piece = extract_sse_text(line[5:])
                if piece:
                    yield piece

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        raw = self._last_user_text(frame.context)
        query = raw
        if self._memory is not None and raw:
            try:
                query = await self._memory.build_query(raw)
            except Exception as e:  # noqa: BLE001 -- memory must never break a turn
                logger.warning(f"build_query failed ({type(e).__name__}); using raw")
                query = raw

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        answer = ""
        try:
            async for piece in self._stream_chain(query):
                answer += piece
                await self.push_frame(LLMTextFrame(piece))
        except httpx.TimeoutException as e:
            await self.push_error(error_msg="weather chain timeout", exception=e)
            answer = "抱歉，天氣服務反應太慢。"  # timeout fallback
            await self.push_frame(LLMTextFrame(answer))
        except Exception as e:  # noqa: BLE001
            await self.push_error(error_msg=f"weather chain error: {type(e).__name__}", exception=e)
            answer = "抱歉，天氣服務暫時連線不上。"  # connect fallback
            await self.push_frame(LLMTextFrame(answer))
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

        if self._memory is not None and raw:
            try:
                self._memory.record_turn(raw, answer)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"record_turn failed ({type(e).__name__})")
