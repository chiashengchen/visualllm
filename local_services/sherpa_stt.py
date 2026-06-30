"""Pipecat STREAMING STT using sherpa-onnx (in-process, CPU, ~0 VRAM).

Unlike the segmented SenseVoice service, this is a TRUE streaming STT: each audio chunk
is fed to a sherpa-onnx OnlineRecognizer with built-in endpoint detection, so it does
NOT depend on the transport's energy-VAD firing (which was unreliable on this box).

Crucially, this service DRIVES TURN-TAKING from the ASR endpoint detector: it emits
VADUserStartedSpeakingFrame when recognized speech begins and VADUserStoppedSpeakingFrame
(+ the final TranscriptionFrame) when sherpa detects end-of-utterance. The downstream
LLMUserAggregator flushes the user turn on VADUserStoppedSpeakingFrame, so this works even
when the energy-VAD never fires (e.g. quiet/attenuated mic audio). Endpoint detection runs
on the acoustic model output, not an energy threshold, so it is robust to low input level.

Bilingual zh-en; zh output is converted to Traditional (zh-TW) via OpenCC (s2twp).
Model: sherpa-onnx streaming zipformer bilingual (k2-fsa), int8, downloaded under models/.
"""
from __future__ import annotations

import os
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601


def _find(model_dir: str, *names: str) -> str:
    for n in names:
        p = os.path.join(model_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {names} found in {model_dir}")


class SherpaStreamingSTTService(STTService):
    def __init__(self, *, model_dir: str, to_traditional: bool = True,
                 endpoint_silence: float = 0.5, sample_rate: int | None = None, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        import sherpa_onnx

        # int8 encoder/joiner = much smaller + faster on CPU at negligible accuracy cost.
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=_find(model_dir, "tokens.txt"),
            encoder=_find(model_dir, "encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx"),
            decoder=_find(model_dir, "decoder-epoch-99-avg-1.onnx", "decoder-epoch-99-avg-1.int8.onnx"),
            joiner=_find(model_dir, "joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx"),
            num_threads=2,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            # rule2 = trailing silence (s) after speech that FIRES the query. The "fire easier"
            # knob (SHERPA_ENDPOINT_SILENCE). rule1 (no speech yet) tracks it so a brief utterance
            # still fires promptly; rule3 caps runaway length.
            rule1_min_trailing_silence=max(endpoint_silence, 1.2),
            rule2_min_trailing_silence=endpoint_silence,
            rule3_min_utterance_length=300,
        )
        self._stream = self._rec.create_stream()
        self._speaking = False
        self._last_partial = ""
        self._cc = None
        if to_traditional:
            import opencc
            self._cc = opencc.OpenCC("s2twp")
        logger.info(f"Sherpa streaming STT ready (model={model_dir}, traditional={to_traditional})")

    def _conv(self, text: str) -> str:
        return self._cc.convert(text) if self._cc else text

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Called per audio frame (streaming). Feed it, decode what's ready, then act on
        # the recognizer's partial result + endpoint state.
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(self.sample_rate, samples)
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        text = self._rec.get_result(self._stream).strip()

        # Speech onset: drive the user turn from the ASR (not the energy-VAD).
        if text and not self._speaking:
            self._speaking = True
            yield VADUserStartedSpeakingFrame()

        if self._rec.is_endpoint(self._stream):
            if text:
                yield TranscriptionFrame(self._conv(text), "", time_now_iso8601())
            if self._speaking:
                yield VADUserStoppedSpeakingFrame()
                self._speaking = False
            self._rec.reset(self._stream)
            self._last_partial = ""
        elif text and text != self._last_partial:
            self._last_partial = text
            yield InterimTranscriptionFrame(self._conv(text), "", time_now_iso8601())
