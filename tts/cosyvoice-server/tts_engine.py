"""
Local TTS engine wrapper around CosyVoice2 (FunAudioLLM).

Loads the model once, registers a fixed reference voice, and exposes a simple
synthesize(text) -> (waveform, sample_rate) API. Used by test_en.py, test_zh.py,
benchmark.py, and app.py so they all share one engine implementation.

Voice: the reference voice is CosyVoice's bundled `asset/zero_shot_prompt.wav`
(a female Mandarin speaker) plus its transcript. Swap PROMPT_WAV / PROMPT_TEXT
to change the forecaster's voice — no other code changes needed.

Language routing: Chinese (and any CJK) text uses inference_zero_shot; Latin /
English text uses inference_cross_lingual (the upstream-recommended path for a
target language that differs from the prompt's language).
"""
from __future__ import annotations

import os
import sys
import re
import logging
from pathlib import Path

# Quiet the HuggingFace tokenizers fork warning (CosyVoice forks after tokenizing).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- locate the cloned CosyVoice repo and put it (and Matcha-TTS) on sys.path -
HERE = Path(__file__).resolve().parent
COSY_DIR = HERE / "CosyVoice"
MATCHA_DIR = COSY_DIR / "third_party" / "Matcha-TTS"
for p in (str(COSY_DIR), str(MATCHA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_MODEL_DIR = str(COSY_DIR / "pretrained_models" / "CosyVoice2-0.5B")

# --- fixed reference voice ---------------------------------------------------
# Overridable via env so the registered voice can be swapped without editing source:
#   COSYVOICE_PROMPT_WAV  = path to the reference clip (clean, mono)
#   COSYVOICE_PROMPT_TEXT = the EXACT transcript of that clip (zero-shot needs it)
# Defaults reproduce the original "weather" female zero-shot speaker.
PROMPT_WAV = os.environ.get(
    "COSYVOICE_PROMPT_WAV", str(COSY_DIR / "asset" / "zero_shot_prompt.wav")
)
PROMPT_TEXT = os.environ.get(
    "COSYVOICE_PROMPT_TEXT", "希望你以后能够做的比我还好呦。"
)
SPK_ID = os.environ.get("COSYVOICE_SPK_ID", "weather")

_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")


def is_cjk(text: str) -> bool:
    """True if the text contains any Chinese/Japanese characters."""
    return bool(_CJK.search(text))


class TTSEngine:
    def __init__(self, model_dir: str | None = None, fp16: bool = False, load_vllm: bool = False):
        import torch
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.model_dir = model_dir or os.environ.get("COSYVOICE_MODEL_DIR", DEFAULT_MODEL_DIR)
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model not found at {self.model_dir}. Download it with:\n"
                f"  python -c \"from modelscope import snapshot_download; "
                f"snapshot_download('iic/CosyVoice2-0.5B', local_dir='{self.model_dir}')\""
            )

        # CosyVoice2 uses CUDA when available, else CPU. MPS is not used by the
        # upstream model, so on Apple Silicon this runs on CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading CosyVoice2 from %s (device=%s)", self.model_dir, self.device)

        # load_vllm: swap the autoregressive LLM onto vLLM (the real fix for first-chunk latency
        # -- the LLM token-gen is the ~3s bottleneck). Off by default (COSYVOICE_VLLM=1 to enable);
        # the Windows server stays on the PyTorch path. Requires the vLLM env (Linux/WSL).
        self.model = CosyVoice2(self.model_dir, load_jit=False, load_trt=False,
                                load_vllm=load_vllm, fp16=fp16)
        self.sample_rate = self.model.sample_rate

        # Register the reference voice once; subsequent calls reuse it by id.
        ok = self.model.add_zero_shot_spk(PROMPT_TEXT, PROMPT_WAV, SPK_ID)
        if ok is not True:
            logging.warning("add_zero_shot_spk returned %r; falling back to per-call prompt", ok)
        self._spk_ready = ok is True

    def synthesize(self, text: str, speed: float = 1.0):
        """Return (waveform_tensor[1, N], sample_rate). Non-streaming."""
        import torch

        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        if is_cjk(text):
            if self._spk_ready:
                gen = self.model.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=SPK_ID, stream=False, speed=speed
                )
            else:
                gen = self.model.inference_zero_shot(
                    text, PROMPT_TEXT, PROMPT_WAV, stream=False, speed=speed
                )
        else:
            # Cross-lingual: English target voiced with the Mandarin reference.
            spk = SPK_ID if self._spk_ready else ""
            gen = self.model.inference_cross_lingual(
                text, PROMPT_WAV, zero_shot_spk_id=spk, stream=False, speed=speed
            )

        chunks = [out["tts_speech"] for out in gen]
        if not chunks:
            raise RuntimeError("CosyVoice produced no audio")
        return torch.concat(chunks, dim=1), self.sample_rate

    def synthesize_stream(self, text: str, speed: float = 1.0):
        """Yield (waveform_tensor[1, N], sample_rate) chunks as they synthesize.

        Same voice/language routing as synthesize(), but stream=True so the first
        chunk is emitted before the whole utterance is done -- the path the realtime
        pipeline (Pipecat -> avatar) needs to start lip-syncing within the TTFO budget.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        if is_cjk(text):
            if self._spk_ready:
                gen = self.model.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=SPK_ID, stream=True, speed=speed
                )
            else:
                gen = self.model.inference_zero_shot(
                    text, PROMPT_TEXT, PROMPT_WAV, stream=True, speed=speed
                )
        else:
            spk = SPK_ID if self._spk_ready else ""
            gen = self.model.inference_cross_lingual(
                text, PROMPT_WAV, zero_shot_spk_id=spk, stream=True, speed=speed
            )

        for out in gen:
            yield out["tts_speech"], self.sample_rate


# Module-level singleton so importing scripts share one loaded model.
_ENGINE: TTSEngine | None = None


def get_engine() -> TTSEngine:
    global _ENGINE
    if _ENGINE is None:
        # NOTE: fp16 was measured to NOT help here (CosyVoice2-0.5B is bottlenecked by
        # autoregressive token generation, not FLOPs -- fp16 RTF was slightly WORSE on the
        # Blackwell GPU). fp32 is the validated-good path; opt into fp16 via COSYVOICE_FP16=1.
        import torch
        fp16 = (os.environ.get("COSYVOICE_FP16", "0").lower() in ("1", "true", "yes", "on")
                and torch.cuda.is_available())
        load_vllm = (os.environ.get("COSYVOICE_VLLM", "0").lower() in ("1", "true", "yes", "on")
                     and torch.cuda.is_available())
        _ENGINE = TTSEngine(fp16=fp16, load_vllm=load_vllm)
    return _ENGINE
