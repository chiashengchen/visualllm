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
ENV=/home/porsche/miniconda3/envs/cosyvllm
export PATH=$ENV/bin:$PATH
export CC=$ENV/bin/gcc
export CXX=$ENV/bin/g++
export COSYVOICE_VLLM=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
# VRAM trim (2026-06-30, measured): cap max sequence length + the card fraction vLLM may use.
# CosyVoice generates ONE short sentence of speech tokens per request, so the default KV
# reservation (max_model_len 32768) is wildly oversized. Capping max-len to 2048 lets the
# util fraction drop far below the old ~0.25 "floor": at 0.16 -> vLLM ~3.7GB with 74x KV
# headroom; at 0.12 -> ~3.4GB / 47x (verified: en + a 27s zh paragraph synth clean, no
# truncation). 0.16 is the robust default; set COSYVOICE_VLLM_GPU_UTIL=0.12 to squeeze the
# whole stack under 8GB. Lower util also reserves less of the shared card -> friendlier to
# the MuseTalk load-order (less "No available memory for the cache blocks"). Override either.
export COSYVOICE_VLLM_MAX_LEN=${COSYVOICE_VLLM_MAX_LEN:-2048}
export COSYVOICE_VLLM_GPU_UTIL=${COSYVOICE_VLLM_GPU_UTIL:-0.16}
cd /mnt/e/Claude/cosyvoice-local-tts
exec $ENV/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001
