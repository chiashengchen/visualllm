"""BLOCKING quality gate for the MuseTalk TensorRT render path.

Renders the SAME audio segment through the PyTorch and TensorRT paths and
compares the composited frames with SSIM. PASS iff min SSIM >= 0.99 (fp16 TRT
must be visually identical to fp16 torch — we do not ship faster-but-worse lips).

Run from the repo root in the `musetalk` env (engines must be built first):
    set PYTHONPATH=. & E:\\miniconda3\\envs\\musetalk\\python.exe -m scripts._trt_quality_check
Verified 2026-06-30: min SSIM 1.0000 (identical). Needs scikit-image.
"""
import os
import numpy as np

os.environ["MUSETALK_TRT"] = "0"   # don't auto-init in load(); we toggle manually

from local_services.musetalk_server.app import engine, SEG_FRAMES

engine.load()
engine._init_trt()                 # build the TRT dict (engines must exist on disk)

from skimage.metrics import structural_similarity as ssim

trt_dict = engine._trt
seg = (0.05 * np.sin(np.arange(engine.samples_for_frames(SEG_FRAMES)) / 30.0)).astype(np.float32)
sz = engine.size


def _render(use_trt):
    engine._trt = trt_dict if use_trt else None
    engine.idx = 0
    return engine.render_segment(seg.copy())


trt_frames = _render(True)
torch_frames = _render(False)
n = min(len(trt_frames), len(torch_frames))
scores = [
    ssim(
        np.frombuffer(trt_frames[i], np.uint8).reshape(sz, sz, 3),
        np.frombuffer(torch_frames[i], np.uint8).reshape(sz, sz, 3),
        channel_axis=2,
    )
    for i in range(n)
]
print("frames %d | min SSIM %.4f | mean SSIM %.4f" % (n, min(scores), sum(scores) / n))
print("RESULT:", "PASS" if min(scores) >= 0.99 else "FAIL")
