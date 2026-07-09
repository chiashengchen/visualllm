"""Standalone check for FunasrSTTService with a mocked server. Run:
    python -m local_services._funasr_stt_check
No network/model needed."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import TranscriptionFrame

from local_services.funasr_stt import FunasrSTTService


def _fake_post(text):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value={"text": text})
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post = MagicMock(return_value=cm)
    return session


async def _collect(svc):
    out = []
    async for f in svc.run_stt(b"\x00\x00" * 100):
        if f is not None:
            out.append(f)
    return out


def test_emits_transcription():
    svc = FunasrSTTService(base_url="http://x")
    with patch.object(svc, "_get_session", AsyncMock(return_value=_fake_post("天氣晴朗"))):
        frames = asyncio.run(_collect(svc))
    assert len(frames) == 1 and isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "天氣晴朗"
    print("emits TranscriptionFrame OK")


def test_failure_yields_nothing():
    svc = FunasrSTTService(base_url="http://x")
    failing = AsyncMock(side_effect=RuntimeError("down"))
    with patch.object(svc, "_get_session", failing):
        frames = asyncio.run(_collect(svc))
    assert frames == []
    print("graceful-degrade OK")


if __name__ == "__main__":
    test_emits_transcription()
    test_failure_yields_nothing()
    print("all checks OK")
