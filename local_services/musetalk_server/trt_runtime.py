"""Minimal TensorRT runtime: load a serialized engine and run it on torch CUDA
tensors (zero-copy via data_ptr binding) on the current torch CUDA stream.

One module, one responsibility (engine load + execution). app.py uses it via
MuseTalkEngine; it could be swapped for torch_tensorrt internally without
touching the server. Inputs are torch CUDA tensors the caller already owns;
outputs are freshly-allocated torch CUDA tensors returned by output name.
"""
from __future__ import annotations

import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
}


class TRTModule:
    def __init__(self, engine_path, device):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, **kw) -> dict:
        for n in self.inputs:
            t = kw[n].contiguous()
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            dt = _TRT_TO_TORCH[self.engine.get_tensor_dtype(n)]
            o = torch.empty(shape, dtype=dt, device=self.device)
            outs[n] = o
            self.ctx.set_tensor_address(n, o.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT execute_async_v3 returned False")
        stream.synchronize()
        return outs
