"""Standalone smoke check for the SenseVoice server helpers. Run with the system python
(tests the pure helper) or post a wav to a running :8004 server.

    python -m local_services.funasr_server._smoke           # unit: PCM conversion
"""
import numpy as np

from local_services.funasr_server.app import _pcm16_to_float32, _strip_emoji


def test_pcm_conversion():
    pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
    out = _pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out.shape == (4,)
    assert abs(out[1] - 0.5) < 1e-3 and abs(out[2] + 0.5) < 1e-3
    print("PCM conversion OK:", out)


def test_strip_emoji():
    # SenseVoice rich postprocess injects emotion emojis; they must not reach the LLM.
    assert _strip_emoji("你好\U0001f60a") == "你好"
    assert _strip_emoji("hello \U0001f3bc world") == "hello  world"
    assert "\U0001f60a" not in _strip_emoji("test\U0001f60a")
    print("strip_emoji OK")


if __name__ == "__main__":
    test_pcm_conversion()
    test_strip_emoji()
    print("smoke OK")
