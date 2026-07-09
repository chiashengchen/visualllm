"""Standalone check: drive SherpaStreamingSTTService with a real PCM clip, chunk-by-chunk
like the pipeline does, and assert it emits VADUserStarted -> Transcription -> VADUserStopped.
    python -m local_services._sherpa_stt_check
Needs the model under models/ and the system python (sherpa-onnx + opencc)."""
import asyncio

from pipecat.frames.frames import (
    StartFrame, TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from local_services.sherpa_stt import SherpaStreamingSTTService


async def main():
    svc = SherpaStreamingSTTService(
        model_dir="models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    )
    svc._sample_rate = 16000  # normally set from the StartFrame's audio_in_sample_rate

    pcm = open("output/_stt_test_16k.pcm", "rb").read()
    # append 2s of silence so the endpoint detector fires (the live transport streams
    # continuous audio after the user stops, so trailing silence accumulates the same way)
    pcm += b"\x00\x00" * 32000

    started = stopped = False
    transcript = None
    # feed in 3200-byte (1600-sample = 100ms) chunks
    for i in range(0, len(pcm), 3200):
        async for f in svc.run_stt(pcm[i:i + 3200]):
            if isinstance(f, VADUserStartedSpeakingFrame):
                started = True
            elif isinstance(f, TranscriptionFrame):
                transcript = f.text
            elif isinstance(f, VADUserStoppedSpeakingFrame):
                stopped = True
    print("VADUserStartedSpeakingFrame emitted:", started)
    print("TranscriptionFrame text:", transcript)
    print("VADUserStoppedSpeakingFrame emitted:", stopped)
    assert started, "no speech-start frame"
    assert transcript, "no transcript"
    assert stopped, "no speech-stop frame (turn would never flush)"
    print("OK: streaming STT drives the turn end-to-end")


if __name__ == "__main__":
    asyncio.run(main())
