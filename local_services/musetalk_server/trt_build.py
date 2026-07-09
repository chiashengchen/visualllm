"""Build a fp16 TensorRT engine from an ONNX file via the TRT Python Builder API.

We use the Python Builder/OnnxParser (NOT the trtexec binary) because the pip
`tensorrt-cu12` wheel does not ship trtexec on Windows. The optimization profile
covers the batch range (min..max) so one engine serves the turn-start partial
segment, full mid-turn batches, and the speech_end tail.
"""
from __future__ import annotations


def build_engine(onnx_path, engine_path, profiles, workspace_mb: int = 2048) -> str:
    """profiles: {input_name: (min_shape, opt_shape, max_shape)} (tuples)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, str(parser.get_error(i)))
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    profile = builder.create_optimization_profile()
    for name, (mn, op, mx) in profiles.items():
        profile.set_shape(name, mn, op, mx)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build returned None: {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return str(engine_path)


def build_all(workspace_mb: int = 2048) -> dict:
    """One-shot: export both realtime models to ONNX and compile fp16 engines.

    Wraps the four SETUP.md one-liners so a fresh machine builds the engines with
    ONE command (`python -m local_services.musetalk_server.trt_build`). Engines are
    GPU/driver-specific, so this must run on the target box in the `musetalk` env;
    rebuild after any GPU/driver change. ~7 min.

    The build FORCES the PyTorch render path (`MUSETALK_TRT=0`) before loading the
    server engine, because the UNet input-capture runs render_segment -- if the TRT
    path were active it would try to use the very engines we are (re)building.
    """
    import os
    import time

    os.environ["MUSETALK_TRT"] = "0"   # capture must run the torch path, not stale/absent engines

    # Imported here (not at module top) so `build_engine` stays importable without
    # dragging in the heavy MuseTalk stack for the existing one-liner callers.
    from local_services.musetalk_server.app import engine, SERVER_DIR, BATCH_SIZE
    from local_services.musetalk_server.trt_export import export_unet_onnx, export_vae_onnx

    cache = SERVER_DIR / "trt_cache"
    cache.mkdir(parents=True, exist_ok=True)

    print("[trt_build] loading MuseTalk models (PyTorch path) ...")
    t0 = time.time()
    engine.load()
    B = int(BATCH_SIZE)   # opt/max batch = the server's segment batch; min=1 (turn-start partial)

    print("[trt_build] 1/2 UNet: torch -> ONNX ...")
    unet_onnx = cache / "unet.onnx"
    meta = export_unet_onnx(engine, unet_onnx)
    S = int(meta["audio"][1])   # audio cross-attn seq len, captured from a real segment (was hardcoded 50)
    print(f"[trt_build]     UNet: ONNX -> fp16 engine (batch 1..{B}, audio seq {S}) ...")
    build_engine(
        unet_onnx, cache / "unet.engine",
        {
            "latent": ((1, 8, 32, 32), (B, 8, 32, 32), (B, 8, 32, 32)),
            "audio": ((1, S, 384), (B, S, 384), (B, S, 384)),
            "timestep": ((1,), (1,), (1,)),
        },
        workspace_mb=workspace_mb,
    )

    print("[trt_build] 2/2 VAE decoder: torch -> ONNX ...")
    vae_onnx = cache / "vae.onnx"
    export_vae_onnx(engine, vae_onnx)
    print(f"[trt_build]     VAE: ONNX -> fp16 engine (batch 1..{B}) ...")
    build_engine(
        vae_onnx, cache / "vae.engine",
        {"latent": ((1, 4, 32, 32), (B, 4, 32, 32), (B, 4, 32, 32))},
        workspace_mb=workspace_mb,
    )

    print(f"[trt_build] done in {time.time() - t0:.0f}s. Engines in {cache} "
          f"(set MUSETALK_TRT=1 to use them).")
    return {"unet": str(cache / "unet.engine"), "vae": str(cache / "vae.engine")}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build the MuseTalk TensorRT engines (UNet + VAE) for MUSETALK_TRT=1."
    )
    ap.add_argument("--workspace-mb", type=int, default=2048,
                    help="TensorRT builder workspace pool cap (MB).")
    args = ap.parse_args()
    build_all(workspace_mb=args.workspace_mb)
