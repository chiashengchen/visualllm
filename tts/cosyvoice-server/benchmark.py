"""
Phase 4 — Benchmark CosyVoice2 latency on this machine.

Measures, per sample: generation time, audio duration, and real-time factor
(RTF = generation_time / audio_duration). RTF < 1 means faster than real time.

A warmup run is done first (the first synthesis pays one-time model warmup /
graph init costs) and excluded from the reported numbers.
"""
import time
import statistics
import torchaudio
import os
from tts_engine import get_engine

SAMPLES = [
    ("en", "Today's weather in Taipei is sunny."),
    ("zh", "今天台北天氣晴朗，氣溫二十八度。"),
    ("zh_long", "各位觀眾早安，今天北台灣多雲時晴，午後山區有短暫陣雨，"
                "氣溫攝氏二十二到二十九度，外出記得攜帶雨具。"),
]
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")


def measure(engine, text):
    t0 = time.perf_counter()
    wav, sr = engine.synthesize(text)
    gen = time.perf_counter() - t0
    dur = wav.shape[1] / sr
    return gen, dur, dur / gen if gen else 0, wav, sr


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    engine = get_engine()

    print("Warmup...")
    engine.synthesize("warm up the model")

    print(f"\n{'sample':<10}{'gen(s)':>9}{'audio(s)':>10}{'RTF':>8}")
    print("-" * 37)
    rtfs = []
    for name, text in SAMPLES:
        gen, dur, rtf, wav, sr = measure(engine, text)
        rtfs.append(gen / dur if dur else 0)
        torchaudio.save(os.path.join(OUTDIR, f"bench_{name}.wav"), wav, sr)
        print(f"{name:<10}{gen:>9.2f}{dur:>10.2f}{rtf:>8.2f}")

    print("-" * 37)
    print(f"\nMean RTF (gen/audio): {statistics.mean(rtfs):.2f}  "
          f"(<1.0 = real-time capable)")
    print(f"Device: {engine.device}   Sample rate: {engine.sample_rate} Hz")
    print("\nExample line:")
    g, d, r, _, _ = measure(engine, SAMPLES[1][1])
    print(f"  Generation Time: {g:.2f} seconds")
    print(f"  Audio Length:    {d:.2f} seconds")
    print(f"  RTF:             {d / g:.2f}")


if __name__ == "__main__":
    main()
