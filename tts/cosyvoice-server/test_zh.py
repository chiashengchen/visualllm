"""Phase 3 — Mandarin (zh-TW) minimum working example. -> outputs/output_zh.wav"""
import os
import torchaudio
from tts_engine import get_engine

TEXT = "今天台北天氣晴朗，氣溫二十八度。"
OUT = os.path.join(os.path.dirname(__file__), "outputs", "output_zh.wav")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    engine = get_engine()
    wav, sr = engine.synthesize(TEXT)
    torchaudio.save(OUT, wav, sr)
    dur = wav.shape[1] / sr
    print(f"[OK] wrote {OUT}  ({dur:.2f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
