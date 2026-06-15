"""A/B the open Thai model against ElevenLabs — the fast answer to "is it good enough?".

Synthesizes the SAME persona lines used in the prototype (prototype-3d-avatar/src/persona.js)
with F5-TTS-THAI and writes them to E:\\Claude\\visualllm-business\\voice-samples\\ next to the
existing ElevenLabs Bella/Matilda samples, for a blind side-by-side ears test.

Run AFTER the model deps are installed (see requirements.txt), in the same isolated venv:
    python -m local_services.f5_thai_server.ab_thai_voice

Note: audio tags like [shy]/[giggles] are eleven_v3-only — they're stripped here, since F5-TTS
carries emotion via the reference clip's prosody, not bracket tags. That asymmetry is exactly
what the ears test is judging.
"""
from __future__ import annotations

# MUST be first: torch-before-f5_tts segfault guard + torchcodec bypass + ffmpeg PATH. See _compat.
from . import _compat  # noqa: F401

import os
import re
import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path(os.getenv("AB_OUT_DIR", r"E:\Claude\visualllm-business\voice-samples"))

# Mirrors SAMPLE_LINES in prototype-3d-avatar/src/persona.js (kept in sync by hand).
LINES = {
    "f5th_greet": "เอ่อ... สวัสดีค่ะ ดีใจจังที่ได้เจอกันอีกแล้ว วันนี้คุณดูดีจังเลยนะคะ",
    "f5th_tease": "แหม... จ้องหน้ากันขนาดนี้ มีอะไรติดหน้าเราหรือเปล่าคะ หรือว่า... แอบมองเพราะเราน่ารัก?",
    "f5th_comfort": "วันนี้เหนื่อยไหมคะ... ไม่เป็นไรนะ เราอยู่ตรงนี้ข้างๆ คุณเสมอ พักผ่อนให้สบายนะคะ",
    "f5th_laugh": "ฮะๆ คุณนี่ตลกจังเลยค่ะ! คุยด้วยทีไรเรากลั้นขำไม่ได้สักที มีความสุขที่สุดเลย",
}

_strip_tags = lambda s: re.sub(r"\[[^\]]*\]", " ", s).strip()


def _save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    pcm = (np.clip(np.asarray(wav, np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> None:
    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS

    repo = os.getenv("F5_THAI_REPO", "VIZINTZOR/F5-TTS-THAI")
    ckpt = hf_hub_download(repo, os.getenv("F5_THAI_CKPT", "model_1000000.pt"))
    vocab = hf_hub_download(repo, os.getenv("F5_THAI_VOCAB", "vocab.txt"))
    ref_audio = os.getenv("F5_THAI_REF_AUDIO") or hf_hub_download(repo, "sample/ref_audio.wav")
    ref_txt_file = Path(__file__).with_name("ref_text.txt")
    ref_text = os.getenv("F5_THAI_REF_TEXT") or (
        ref_txt_file.read_text(encoding="utf-8").strip() if ref_txt_file.exists() else ""
    )
    model = F5TTS(model=os.getenv("F5_THAI_ARCH", "F5TTS_v1_Base"), ckpt_file=ckpt, vocab_file=vocab)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in LINES.items():
        wav, sr, _ = model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=_strip_tags(text),
            remove_silence=True,
        )
        out = OUT_DIR / f"{name}.wav"
        _save_wav(out, wav, sr)
        print(f"  wrote {out}")
    print("\nDone. Compare against Bella.mp3 / Matilda.mp3 in the same folder (ears test).")


if __name__ == "__main__":
    main()
