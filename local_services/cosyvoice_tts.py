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

import os
import time
import wave
from pathlib import Path
from typing import AsyncGenerator

import aiohttp
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.tts_service import TTSService

# Live noise/garble detector (COSYVOICE_NOISE_LOG=1, on by default). CosyVoice2 occasionally
# samples a bad token sequence and emits a screech or runaway babble instead of speech. We can't
# reproduce it on demand, so we watch the live stream: each utterance's PCM is analysed and, when
# it looks garbled, we WARN with the exact text + dump the wav to output/ for inspection.
_NOISE_LOG = (os.getenv("COSYVOICE_NOISE_LOG", "1").lower() in ("1", "true", "yes", "on"))
# Capture-EVERY-utterance debug mode: save every reply's wav + log its stats, so an
# intermittent garble that plays at NORMAL volume (no clipping -> amplitude detector misses it)
# can still be pulled by timestamp/text after the user hears it. COSYVOICE_CAPTURE_ALL=1.
_CAPTURE_ALL = (os.getenv("COSYVOICE_CAPTURE_ALL", "0").lower() in ("1", "true", "yes", "on"))
_NOISE_DIR = Path(__file__).resolve().parent.parent / "output" / "cosy_noise"


def _analyze_pcm(pcm: bytes, sr: int, text: str) -> dict:
    """Compute utterance audio stats + whether it looks garbled (always returns a dict)."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if a.size < sr // 10:   # <0.1s -- too short to judge
        return dict(dur=round(a.size / sr, 2), rms=0, peak=0, clip=0, zcr=0,
                    flat=0, bad=False, why="too short")
    norm = a / 32768.0
    dur = a.size / sr
    rms = float(np.sqrt(np.mean(norm ** 2)))
    peak = float(np.max(np.abs(norm)))
    clip = float(np.mean(np.abs(a) >= 32000))                    # near int16 max -> screech/clip
    zcr = float(np.mean(np.abs(np.diff(np.sign(norm))) > 0))      # noise -> high zero-crossing rate
    # Spectral flatness: white-noise/garble ~1, tonal speech ~0. Catches normal-VOLUME garble
    # that amplitude thresholds miss (the case the user hears: screech at normal loudness).
    seg = norm[: (len(norm) // 1024) * 1024]
    flat = 0.0
    if seg.size >= 1024:
        spec = np.abs(np.fft.rfft(seg.reshape(-1, 1024), axis=1)) ** 2 + 1e-10
        gm = np.exp(np.mean(np.log(spec), axis=1))
        am = np.mean(spec, axis=1)
        flat = float(np.mean(gm / am))
    est = max(0.4, len(text) * 0.09)                             # rough speech-duration estimate
    why = []
    if clip > 0.02:                 why.append(f"clip {clip*100:.1f}%")
    if rms > 0.45:                  why.append(f"loud rms {rms:.2f}")
    if flat > 0.3:                  why.append(f"noise-like flat {flat:.2f}")
    if zcr > 0.45 and rms > 0.2:    why.append(f"noisy zcr {zcr:.2f}")
    if dur > est * 3.5:             why.append(f"too long {dur:.1f}s vs ~{est:.1f}s (babble?)")
    return dict(dur=round(dur, 2), rms=round(rms, 3), peak=round(peak, 3), clip=round(clip, 4),
                zcr=round(zcr, 3), flat=round(flat, 3), bad=bool(why), why=", ".join(why) or "ok")


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
                parts: list[bytes] = [] if _NOISE_LOG else None
                async for chunk in resp.content.iter_chunked(chunk_bytes):
                    if not chunk:
                        continue
                    if first:
                        await self.stop_ttfb_metrics()
                        first = False
                    if parts is not None:
                        parts.append(chunk)
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                if parts is not None:
                    self._check_noise(b"".join(parts), text)
        except Exception as e:  # noqa: BLE001
            logger.exception("CosyVoice TTS failed")
            yield ErrorFrame(f"CosyVoice TTS error: {e}")

    def _check_noise(self, pcm: bytes, text: str) -> None:
        """Flag + dump garbled utterances (and, in capture-all mode, EVERY utterance) for
        offline inspection. Best-effort -- diagnostics must never break TTS."""
        try:
            res = _analyze_pcm(pcm, self.sample_rate, text)
            save = _CAPTURE_ALL or res["bad"]
            path_name = ""
            if save:
                _NOISE_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                safe = "".join(c if c.isalnum() else "_" for c in text[:30]).strip("_")
                tag = "BAD_" if res["bad"] else ""
                path = _NOISE_DIR / f"{stamp}_{tag}{safe or 'utt'}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(self.sample_rate)
                    w.writeframes(pcm)
                path_name = f" | saved {path.name}"
            stats = (f"dur={res['dur']}s rms={res['rms']} peak={res['peak']} "
                     f"clip={res['clip']} zcr={res['zcr']} flat={res['flat']}")
            if res["bad"]:
                logger.warning(f"[cosy noise?] {res['why']} | {stats} | text=[{text}]{path_name}")
            elif _CAPTURE_ALL:
                logger.info(f"[cosy ok] {stats} | text=[{text}]{path_name}")
        except Exception:  # noqa: BLE001 -- diagnostics must never break TTS
            pass
