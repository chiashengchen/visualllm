"""FP8 (E4M3) post-training quantization of the MuseTalk realtime UNet -> TensorRT (Lever 2a).

WHY: at turn start CosyVoice's opening vocoder burst starves MuseTalk's FIRST render
segment on the shared GPU -- the one thing TRT's mid-turn headroom does NOT cover (see
CLAUDE.md / docs P15). FP8 halves the UNet's GPU compute, so that starved first segment
finishes sooner. The UNet is the compute hog (~168ms of the ~182ms TRT render); the VAE
decoder is tiny and quality-sensitive, so it stays FP16.

FLOW: load the torch render path -> capture REAL (latent,timestep,audio) activations by
driving a real reply WAV -> modelopt FP8 PTQ (calibrated on those activations) -> export
Q/DQ ONNX -> build an FP8+FP16 TensorRT engine -> VALIDATE quality (SSIM of the actual
composited frames vs the FP16 engine) + per-segment GPU time. Writes unet_fp8.engine next
to the fp16 engines; wiring it in is a separate, reversible copy step (see --install).

Run in the `musetalk` env from the repo root, GPU free:
    E:\miniconda3\envs\musetalk\python.exe -m local_services.musetalk_server.trt_quant_fp8 --wav output/_mic_drive.wav
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def _load_wav_16k(path: Path) -> np.ndarray:
    """Read a WAV as float32 mono @16k (the server's audio contract)."""
    import wave

    with wave.open(str(path), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        dst = np.linspace(0, a.shape[0] - 1, num=int(round(a.shape[0] * 16000 / sr)))
        a = np.interp(dst, np.arange(a.shape[0]), a).astype(np.float32)
    return a


def _capture_activations(engine, wav: np.ndarray, max_batches: int = 96):
    """Drive the real WAV through the torch render path, hooking the UNet to grab real
    (latent, timestep, audio) input triples -- the calibration + validation set."""
    import torch

    grabbed: list[tuple] = []
    orig = engine.unet.model.forward

    def hook(latent, timestep, encoder_hidden_states=None, **kw):
        if len(grabbed) < max_batches:
            ts = timestep if torch.is_tensor(timestep) else torch.tensor([0], device=engine.device)
            grabbed.append((latent.detach().clone(), ts.detach().clone(),
                            encoder_hidden_states.detach().clone()))
        return orig(latent, timestep, encoder_hidden_states=encoder_hidden_states, **kw)

    engine.unet.model.forward = hook
    try:
        seg = engine.samples_for_frames(8)
        for i in range(0, len(wav), seg):
            chunk = wav[i:i + seg]
            if len(chunk) < seg:
                chunk = np.concatenate([chunk, np.zeros(seg - len(chunk), np.float32)])
            engine.render_segment(chunk)
            if len(grabbed) >= max_batches:
                break
    finally:
        engine.unet.model.forward = orig
        engine.reset_idx()
    if not grabbed:
        raise RuntimeError("no UNet activations captured (render produced nothing).")
    return grabbed


class _CalibReader:
    """onnxruntime CalibrationDataReader over the real captured UNet input triples."""

    def __init__(self, calib):
        self._items = [
            {
                "latent": l.cpu().numpy(),
                "timestep": t.cpu().numpy().astype(np.int64),
                "audio": a.cpu().numpy(),
            }
            for (l, t, a) in calib
        ]
        self._it = iter(self._items)

    def get_next(self):
        return next(self._it, None)

    def get_first(self):
        return self._items[0]

    def rewind(self):
        self._it = iter(self._items)


def _force_per_tensor_weights():
    """Monkey-patch modelopt's FP8 path to emit PER-TENSOR weight scales instead of its
    hardcoded per_channel=True (fp8.py:302). TensorRT's FP8 CONVOLUTION tactics require
    per-tensor weight scales; the per-channel f16[N] scales modelopt produces are what TRT
    rejects with "No matching rules found for input operand types" -> silent slow fallback.
    We wrap the `quantize_static` symbol in the fp8 module namespace, forcing per_channel=False."""
    import modelopt.onnx.quantization.fp8 as f8

    orig = f8.quantize_static

    def patched(*a, **k):
        k["per_channel"] = False
        return orig(*a, **k)

    f8.quantize_static = patched
    return lambda: setattr(f8, "quantize_static", orig)


def quantize_onnx_fp8(base_onnx: Path, calib, out_onnx: Path, per_tensor: bool = False) -> None:
    """FP8 PTQ at the ONNX graph level (modelopt.onnx). Operates on the already-working
    fp16 UNet ONNX + real calibration activations, so it avoids torch 2.11's TorchScript
    FP8 symbolic issue. Inserts QuantizeLinear/DequantizeLinear TRT consumes as FP8.

    per_tensor=True forces per-tensor weight scales (the fix for TRT FP8 conv rejecting
    modelopt's default per-channel weights)."""
    from modelopt.onnx.quantization import quantize

    restore = _force_per_tensor_weights() if per_tensor else (lambda: None)
    gran = "per-TENSOR weights" if per_tensor else "per-channel weights (modelopt default)"
    print(f"[fp8] modelopt ONNX PTQ (fp8, max calib, {gran}) over {len(calib)} batches -> {out_onnx.name} ...")
    try:
        quantize(
            str(base_onnx),
            quantize_mode="fp8",
            calibration_method="max",   # FP8's proper calibration: per-tensor absolute max, ONE pass.
            #                             (the default 'entropy'/histogram is the slow INT8 method.)
            calibration_data_reader=_CalibReader(calib),
            calibration_eps=["cuda:0"],   # onnxruntime-gpu; fp16 graph needs the CUDA EP
            output_path=str(out_onnx),
        )
    finally:
        restore()


def build_fp8_engine(onnx_path: Path, engine_path: Path, B: int, S: int, workspace_mb: int = 3072):
    """Build a TensorRT engine with BOTH FP8 and FP16 enabled: TRT runs the Q/DQ'd layers
    in FP8 and everything else in FP16."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, str(parser.get_error(i)))
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.FP8)   # <-- the Lever-2a change
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    profile = builder.create_optimization_profile()
    profile.set_shape("latent", (1, 8, 32, 32), (B, 8, 32, 32), (B, 8, 32, 32))
    profile.set_shape("audio", (1, S, 384), (B, S, 384), (B, S, 384))
    profile.set_shape("timestep", (1,), (1,), (1,))
    config.add_optimization_profile(profile)

    print(f"[fp8] building FP8+FP16 engine (batch 1..{B}, audio seq {S}) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("FP8 engine build returned None")
    engine_path.write_bytes(serialized)
    print(f"[fp8] engine written: {engine_path} ({engine_path.stat().st_size/1e6:.0f} MB)")


def validate(engine, cache: Path, calib):
    """Compare the FP8 UNet engine vs the FP16 one: SSIM of the real composited frames +
    per-8-frame-segment GPU time. Runs both engines in-process on the same inputs."""
    import torch
    from local_services.musetalk_server.trt_runtime import TRTModule

    unet_fp16 = TRTModule(str(cache / "unet.engine"), engine.device)
    unet_fp8 = TRTModule(str(cache / "unet_fp8.engine"), engine.device)
    vae = TRTModule(str(cache / "vae.engine"), engine.device)

    def _frames(unet_mod, latent, ts, audio):
        """UNet -> VAE -> composited RGB frames (mirrors app.py render, TRT path)."""
        sample = unet_mod(latent=latent, timestep=ts, audio=audio)["sample"]
        dec_in = (1.0 / engine.vae.scaling_factor) * sample.to(engine.vae.vae.dtype)
        img = vae(latent=dec_in)["image"]
        img = (img / 2 + 0.5).clamp(0, 1)
        recon = (img.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype("uint8")[..., ::-1]
        outs = []
        for k in range(recon.shape[0]):
            engine.idx = k
            outs.append(np.frombuffer(engine._composite(recon[k], k % len(engine.latent_cycle)),
                                      dtype=np.uint8))
        return np.stack(outs).astype(np.float32)

    # quality: SSIM over the real captured batches
    try:
        from skimage.metrics import structural_similarity as ssim
        have_ssim = True
    except Exception:
        have_ssim = False

    ssims, maxdiffs = [], []
    for (latent, ts, audio) in calib[:16]:
        latent16, audio16 = latent.half(), audio.half()
        a = _frames(unet_fp16, latent16, ts, audio16)
        b = _frames(unet_fp8, latent16, ts, audio16)
        maxdiffs.append(float(np.abs(a - b).max()))
        if have_ssim:
            for fa, fb in zip(a, b):
                ssims.append(ssim(fa, fb, channel_axis=None, data_range=255.0)
                             if fa.ndim == 1 else ssim(fa, fb, data_range=255.0))
    engine.reset_idx()

    # speed: GPU time per batch, both engines
    def _bench(mod, n=60):
        latent, ts, audio = calib[0]
        latent, audio = latent.half(), audio.half()
        for _ in range(5):
            mod(latent=latent, timestep=ts, audio=audio)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            mod(latent=latent, timestep=ts, audio=audio)
        torch.cuda.synchronize()
        return 1000 * (time.time() - t0) / n

    ms16, ms8 = _bench(unet_fp16), _bench(unet_fp8)
    return {
        "ssim_mean": float(np.mean(ssims)) if ssims else None,
        "ssim_min": float(np.min(ssims)) if ssims else None,
        "max_pixel_diff": max(maxdiffs) if maxdiffs else None,
        "unet_ms_fp16": ms16, "unet_ms_fp8": ms8,
        "speedup": ms16 / ms8 if ms8 else None,
    }


def main():
    ap = argparse.ArgumentParser(description="FP8 PTQ of the MuseTalk UNet -> TRT (Lever 2a).")
    ap.add_argument("--wav", default="output/_mic_drive.wav", help="real reply WAV for calibration")
    ap.add_argument("--opset", type=int, default=19)
    ap.add_argument("--per-tensor", action="store_true",
                    help="force per-tensor weight scales (TRT FP8-conv needs this; modelopt defaults per-channel)")
    args = ap.parse_args()

    os.environ["MUSETALK_TRT"] = "0"   # capture must run the torch path
    wav_path = Path(args.wav).resolve()   # resolve NOW: engine.load() chdir's into the vendor dir
    from local_services.musetalk_server.app import engine, SERVER_DIR, BATCH_SIZE

    cache = SERVER_DIR / "trt_cache"
    print("[fp8] loading MuseTalk (torch path) ...")
    engine.load()
    B = int(BATCH_SIZE)

    wav = _load_wav_16k(wav_path)
    print(f"[fp8] calibration WAV {args.wav}: {len(wav)/16000:.1f}s")
    calib = _capture_activations(engine, wav)
    S = int(calib[0][2].shape[1])   # audio cross-attn seq len
    print(f"[fp8] captured {len(calib)} UNet batches (audio seq {S}).")

    base_onnx = cache / "unet.onnx"   # the existing fp16 UNet ONNX (base for FP8 PTQ)
    if not base_onnx.exists():
        from local_services.musetalk_server.trt_export import export_unet_onnx
        print("[fp8] base unet.onnx missing; exporting fresh ...")
        export_unet_onnx(engine, base_onnx)
    onnx_path = cache / "unet_fp8.onnx"
    quantize_onnx_fp8(base_onnx, calib, onnx_path, per_tensor=args.per_tensor)
    build_fp8_engine(onnx_path, cache / "unet_fp8.engine", B, S)

    print("[fp8] validating quality + speed vs fp16 ...")
    res = validate(engine, cache, calib)
    print("\n================ FP8 RESULT ================")
    print(f"  UNet GPU/batch : fp16 {res['unet_ms_fp16']:.1f}ms -> fp8 {res['unet_ms_fp8']:.1f}ms "
          f"({res['speedup']:.2f}x)")
    print(f"  frame SSIM     : mean {res['ssim_mean']} min {res['ssim_min']}")
    print(f"  max pixel diff : {res['max_pixel_diff']} / 255")
    print("===========================================")
    print("Engine at trt_cache/unet_fp8.engine. To USE it (reversible):")
    print("  cp unet.engine unet_fp16.engine  &&  cp unet_fp8.engine unet.engine")


if __name__ == "__main__":
    main()
