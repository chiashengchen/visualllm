"""Deterministic proof of the steady-mode 'screech' root cause + fix (no GPU/STT/browser).

ROOT CAUSE (proven 2026-06-22 by byte-diffing the live captures: a 1049-byte deletion at
6.040s, speech otherwise bit-identical -> the rest of the turn went odd-byte-misaligned ->
broadband noise): pipecat's output transport (`MediaSender._next_frame`) calls
`_bot_stopped_speaking()` whenever no audio frame reaches its queue within
`BOT_VAD_STOP_FALLBACK_SECS` (default 3s), and that handler does
`self._audio_buffer = bytearray()` -- DISCARDING the partial audio buffer. In steady/non-live
sync the voice is released paced to RENDERED video, so a >3s render stall starves the queue, the
timeout fires MID-TURN, and the discarded partial buffer (an odd byte count) leaves every later
int16 sample misaligned = the screech.

This drives the REAL `MediaSender` audio loop with a controllable timeout and asserts:
  * default (short) timeout  -> a mid-turn audio gap DISCARDS the buffered partial audio (the bug)
  * raised   timeout         -> the same gap does NOT discard it (the fix in
                                main.py::_relax_bot_vad_stop_timeout)

Run:  python -m scripts._screech_repro_test
"""
import asyncio

from pipecat.transports import base_output
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.frames.frames import OutputAudioRawFrame

MediaSender = BaseOutputTransport.MediaSender


class FakeTransport:
    def create_task(self, coro):
        coro.close()              # don't run the infinite handler loops
        return None

    def get_event_loop(self):
        return asyncio.get_running_loop()

    async def write_audio_frame(self, frame):
        return True

    async def push_frame(self, *a, **k):
        pass


async def _make():
    p = TransportParams(audio_out_enabled=True)
    s = MediaSender(FakeTransport(), destination=None, sample_rate=24000,
                    audio_chunk_size=1920, params=p)
    s._audio_queue = asyncio.Queue()
    # Mid-turn state: the bot IS speaking and a PARTIAL (odd) chunk sits buffered, exactly the
    # window the live bug hit (1049 bytes < the 1920 chunk size, so it isn't emitted yet).
    s._bot_speaking = True
    s._tts_audio_received = True
    await s.handle_audio_frame(
        OutputAudioRawFrame(audio=b"\x11" * 1049, sample_rate=24000, num_channels=1)
    )
    return s


async def _gap_discards_buffer(timeout_secs: float, wait_secs: float) -> bool:
    """Feed nothing for `wait_secs` while the real `_next_frame` consumer runs with
    BOT_VAD_STOP_FALLBACK_SECS = `timeout_secs`. Return whether the partial buffer was discarded."""
    base_output.BOT_VAD_STOP_FALLBACK_SECS = timeout_secs
    s = await _make()
    assert len(s._audio_buffer) == 1049, "precondition: partial buffer present"
    gen = s._next_frame()                       # the REAL transport audio generator

    async def consume():
        async for _frame in gen:                # will block on the empty queue, then time out
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(wait_secs)              # simulate the render-stall gap (no audio arrives)
    discarded = len(s._audio_buffer) == 0
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    return discarded


async def main() -> int:
    # Bug: a 0.2s "default" timeout fires within a 0.5s gap -> buffer discarded.
    bug = await _gap_discards_buffer(timeout_secs=0.2, wait_secs=0.5)
    # Fix: raise the timeout (as _relax_bot_vad_stop_timeout does) -> same gap, buffer kept.
    fixed = not await _gap_discards_buffer(timeout_secs=600.0, wait_secs=0.5)
    print(f"BUG reproduced (mid-turn gap discards partial audio @ short timeout): {bug}")
    print(f"FIX 1 holds (raised timeout keeps the partial audio across the gap):  {fixed}")

    # --- FIX 2: the sample-alignment guard makes ANY buffer-clear harmless ----------------------
    # The deeper invariant: whatever clears the transport buffer (the 3s gap OR the per-turn
    # TTSStoppedFrame), the screech only happens if the discarded byte count is ODD (half-sample
    # shift). musetalk_video._align_even carries a dangling odd byte between downstream frames so
    # every frame -- and thus the transport's running total -- is EVEN. Then a mid-stream clear
    # drops a whole-sample (even) gap = at worst a click, never the screech. Simulate it:
    import os as _os

    def _carry_align(frames):
        """Mirror musetalk_video._align_even: emit even-length frames, carry the odd byte."""
        carry = b""
        for f in frames:
            data = carry + f
            if len(data) & 1:
                carry, data = data[-1:], data[:-1]
            else:
                carry = b""
            yield data

    def _max_odd_remainder(frames, chunk=1920):
        """Feed frames into a transport-style buffer; return the worst (max) ODD remainder seen at
        a clear point (a clear can fire at any frame boundary). Odd remainder => screech possible."""
        buf = bytearray()
        worst = 0
        for f in frames:
            buf.extend(f)
            while len(buf) >= chunk:
                buf = buf[chunk:]
            if len(buf) & 1:
                worst = max(worst, len(buf))
        return worst

    # CosyVoice-like stream: 960B chunks + an ODD final chunk per utterance, several utterances.
    raw_frames = []
    for _ in range(4):
        raw_frames += [_os.urandom(960) for _ in range(5)]
        raw_frames.append(_os.urandom(455))           # odd-length final chunk (the screech seed)
    odd_without = _max_odd_remainder(raw_frames)        # no guard -> odd remainder exists (screech)
    odd_with = _max_odd_remainder(list(_carry_align(raw_frames)))  # guard -> never odd
    guard = odd_without > 0 and odd_with == 0
    print(f"BUG path  (no guard: an ODD buffer remainder can be discarded -> screech): "
          f"{odd_without > 0} (max odd remainder {odd_without}B)")
    print(f"FIX 2 holds (guard: remainder is ALWAYS even -> any clear is a click, not a screech): "
          f"{odd_with == 0}")

    ok = bug and fixed and guard
    print("\nRESULT:", "PASS - root cause + both fixes confirmed" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
