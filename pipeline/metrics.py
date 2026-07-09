"""Latency instrumentation.

TtfoMeter is a Pipecat FrameProcessor you drop into the pipeline. It measures
Time-To-First-Output (TTFO): the gap between the user finishing their turn
(UserStoppedSpeakingFrame) and the bot's first outgoing speech/video
(BotStartedSpeakingFrame). This is the core acceptance metric — target < 3 s.

Place it once near the end of the pipeline so it sees both upstream user events
and downstream bot events.
"""
from __future__ import annotations

import time

from loguru import logger

# Import paths are version-sensitive; keep them isolated here.
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TtfoMeter(FrameProcessor):
    def __init__(self, target_s: float = 3.0):
        super().__init__()
        self._target_s = target_s
        self._turn_end_t: float | None = None
        self.samples: list[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            # User just finished talking — start the clock for this turn. We arm on EITHER
            # frame: Deepgram + the transport's turn machinery emit UserStoppedSpeakingFrame,
            # but the ASR-driven local STT (sherpa) drives turns with VADUserStoppedSpeakingFrame
            # (a SystemFrame, NOT a subclass of UserStoppedSpeakingFrame) — without this the meter
            # never armed on the sherpa path and reported count:0. When both fire (Deepgram), the
            # later one wins; they mark the same instant, so the measurement is unchanged.
            self._turn_end_t = time.monotonic()

        elif isinstance(frame, BotStartedSpeakingFrame) and self._turn_end_t is not None:
            ttfo = time.monotonic() - self._turn_end_t
            self._turn_end_t = None
            self.samples.append(ttfo)
            status = "OK " if ttfo <= self._target_s else "OVER"
            logger.info(f"[TTFO {status}] {ttfo:0.2f}s (target {self._target_s:0.1f}s)")

        # Always forward the frame untouched.
        await self.push_frame(frame, direction)

    def summary(self) -> dict:
        if not self.samples:
            return {"count": 0}
        s = sorted(self.samples)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        return {
            "count": len(s),
            "median_s": round(s[len(s) // 2], 2),
            "p95_s": round(p95, 2),
            "max_s": round(s[-1], 2),
            "target_s": self._target_s,
            "pass": p95 <= self._target_s,
        }
