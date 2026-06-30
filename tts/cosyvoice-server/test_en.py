"""Phase 3 — English minimum working example. -> outputs/output_en.wav"""
import os
import torchaudio
from tts_engine import get_engine

TEXT = "Today's weather in Taipei is sunny."
OUT = os.path.join(os.path.dirname(__file__), "outputs", "output_en.wav")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    engine = get_engine()
    wav, sr = engine.synthesize(TEXT)
    torchaudio.save(OUT, wav, sr)
    dur = wav.shape[1] / sr
    print(f"[OK] wrote {OUT}  ({dur:.2f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
