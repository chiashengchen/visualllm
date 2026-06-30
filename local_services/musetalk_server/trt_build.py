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
