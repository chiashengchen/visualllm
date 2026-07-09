"""ONNX export helpers for the MuseTalk realtime GPU models (UNet + VAE decoder).

Only these two run in the realtime loop (render_segment): the UNet
(diffusers UNet2DConditionModel, latent B,8,32,32 -> sample B,4,32,32) and the
VAE decoder (AutoencoderKL.decode, B,4,32,32 -> image B,3,256,256). We export
each to ONNX so trt_build.py can compile fp16 TensorRT engines. whisper, the
positional encoding, and the PIL compositing stay in PyTorch.

Real-input capture: rather than guess the audio sequence length feeding the
UNet cross-attention, we hook the UNet during one silent segment and grab the
actual (latent, timestep, audio) tensors, so the exported shapes are correct.
"""
from __future__ import annotations

import numpy as np
import torch


class _UNetFwd(torch.nn.Module):
    """ONNX-exportable view of MuseTalk's UNet call:
    render_segment does unet.model(latent, timesteps, encoder_hidden_states=audio).sample"""

    def __init__(self, unet_model):
        super().__init__()
        self.m = unet_model

    def forward(self, latent, timestep, audio):
        return self.m(latent, timestep, encoder_hidden_states=audio).sample


class _VAEDecFwd(torch.nn.Module):
    """Decoder-only view. The caller pre-scales by 1/scaling_factor and does the
    post (clamp/uint8/BGR); the engine covers just vae.decode(...).sample."""

    def __init__(self, vae):
        super().__init__()
        self.v = vae

    def forward(self, latent):
        return self.v.decode(latent).sample


def _capture_unet_inputs(engine) -> dict:
    """Run one silent segment through the torch render path, hooking the UNet to
    grab a real (latent, timestep, audio) triple with the correct shapes/dtypes."""
    grabbed: dict = {}
    orig = engine.unet.model.forward

    def hook(latent, timestep, encoder_hidden_states=None, **kw):
        grabbed.setdefault("latent", latent.detach())
        grabbed.setdefault(
            "timestep",
            timestep.detach() if torch.is_tensor(timestep)
            else torch.tensor([0], device=engine.device),
        )
        grabbed.setdefault("audio", encoder_hidden_states.detach())
        return orig(latent, timestep, encoder_hidden_states=encoder_hidden_states, **kw)

    engine.unet.model.forward = hook
    try:
        engine.render_segment(np.zeros(engine.samples_for_frames(8), dtype=np.float32))
    finally:
        engine.unet.model.forward = orig
        engine.idx = 0
    if "latent" not in grabbed:
        raise RuntimeError("UNet capture failed: the hook never fired (no segment rendered).")
    return grabbed


def export_unet_onnx(engine, out_path) -> dict:
    ins = _capture_unet_inputs(engine)
    wrap = _UNetFwd(engine.unet.model).eval()
    latent, ts, audio = ins["latent"], ins["timestep"], ins["audio"]
    with torch.no_grad():
        torch.onnx.export(
            wrap, (latent, ts, audio), str(out_path),
            input_names=["latent", "timestep", "audio"],
            output_names=["sample"], opset_version=17,
            dynamic_axes={"latent": {0: "B"}, "audio": {0: "B"}, "sample": {0: "B"}},
        )
    return {
        "latent": tuple(latent.shape), "audio": tuple(audio.shape),
        "timestep": tuple(ts.shape), "dtype": str(latent.dtype),
    }


def export_vae_onnx(engine, out_path) -> dict:
    dec = _VAEDecFwd(engine.vae.vae).eval()
    dummy = torch.randn(8, 4, 32, 32, device=engine.device, dtype=engine.vae.vae.dtype)
    with torch.no_grad():
        torch.onnx.export(
            dec, (dummy,), str(out_path),
            input_names=["latent"], output_names=["image"], opset_version=17,
            dynamic_axes={"latent": {0: "B"}, "image": {0: "B"}},
        )
    return {"latent": (8, 4, 32, 32), "dtype": str(dummy.dtype)}
