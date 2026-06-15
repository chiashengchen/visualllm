"""Environment shims that MUST run before importing f5_tts. Import this first.

Three hard-won fixes for the bleeding-edge stack (RTX 5060 Ti / Blackwell, torch 2.11+cu128,
torchaudio 2.11, transformers 5.x) — see README "Gotchas":

1. torch/torchaudio imported here, before f5_tts -> avoids the native load-order segfault.
2. torchaudio 2.11 routes `load` through torchcodec (needs an ffmpeg shared build it can't find
   here). We bypass it with a soundfile-backed `torchaudio.load` monkeypatch.
3. pydub still shells out to the ffmpeg CLI; we add the winget ffmpeg bin to PATH if present.
"""
from __future__ import annotations

import glob
import os
import sys

# torch BEFORE f5_tts (segfault guard) -----------------------------------------
import torch
import torchaudio
import soundfile as _sf

# UTF-8 so Thai prints on the Windows console ----------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ffmpeg CLI on PATH (pydub needs it; torchcodec's shared libs we don't use) ----
if not any(os.path.exists(os.path.join(p, "ffmpeg.exe")) for p in os.environ.get("PATH", "").split(os.pathsep) if p):
    for pat in (
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\bin"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\ffmpeg.exe"),
    ):
        hits = glob.glob(pat, recursive=True)
        if hits:
            bined = hits[0] if os.path.isdir(hits[0]) else os.path.dirname(hits[0])
            os.environ["PATH"] = bined + os.pathsep + os.environ.get("PATH", "")
            break

# torchcodec bypass: soundfile-backed torchaudio.load (returns channels-first float tensor) -----
_orig_load = torchaudio.load


def _soundfile_load(path, *args, **kwargs):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    return torch.from_numpy(data.T).contiguous(), sr                 # (channels, samples)


torchaudio.load = _soundfile_load
