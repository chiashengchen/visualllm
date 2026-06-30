#!/usr/bin/env bash
# Launch CosyVoice2 TTS on vLLM, in WSL (Ubuntu) on the Blackwell 5060 Ti.
# This replaces the Windows PyTorch CosyVoice server on :8001 and cuts first-chunk
# latency ~3.4s -> ~1.2s (vLLM accelerates the autoregressive speech-token LLM).
#
# Why each env var (all required on this bleeding-edge stack -- see the build notes):
#   COSYVOICE_VLLM=1                  -> engine loads the LLM on vLLM (tts_engine.py switch)
#   VLLM_ENABLE_V1_MULTIPROCESSING=0 -> run the engine in-process (no spawn re-import crash)
#   VLLM_USE_FLASHINFER_SAMPLER=0    -> native torch sampler (flashinfer's needs nvcc, not present)
#   CC/CXX + PATH                    -> Triton JITs kernels at runtime; point it at the conda gcc
#   COSYVOICE_VLLM_EAGER=1 (default) -> skip torch.compile/CUDA-graph capture (needs more toolchain)
set -e
# This script lives in the repo at tts/cosyvoice-server/ ; run it from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Path to the `cosyvllm` conda env inside WSL. Override with COSYVLLM_ENV if yours differs.
ENV=${COSYVLLM_ENV:-$HOME/miniconda3/envs/cosyvllm}
export PATH=$ENV/bin:$PATH
export CC=$ENV/bin/gcc
export CXX=$ENV/bin/g++
export COSYVOICE_VLLM=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$SCRIPT_DIR"
exec $ENV/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001
