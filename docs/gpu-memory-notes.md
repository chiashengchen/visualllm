# GPU memory notes — keeping the shared 16GB card from getting too tight

The TTS engine (CosyVoice on vLLM) and the MuseTalk avatar renderer share **one** ~16GB
GPU. This note records how to measure VRAM and the low-risk levers to free some.

## How to measure

```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi   # full table
```
Per-process `used_memory` reads `[N/A]` on Windows/WDDM — you get the process *list* but not
each one's bytes. Compare `memory.free` across states (idle → cosyvoice only → + MuseTalk → mid-turn)
to attribute usage.

## Finding (2026-06-30): much of "used" is the desktop, not the pipeline

A snapshot showed **15796 / 16311 MiB used, 255 MiB free** — but the GPU process list was full of
**non-pipeline apps**: `explorer.exe`, `SearchHost`, `StartMenuExperienceHost`, `msedgewebview2.exe`,
`WindowsTerminal`, `NVIDIA Overlay`, `RazerAppEngine`, `SnippingTool`, `TextInputHost`, `ShellExperienceHost`.

So the card is tight partly because the **Windows desktop + background apps** are GPU clients (the
RDP/desktop compositor, a browser/WebView, vendor overlays). The real-time pipeline itself doesn't need
any of that.

### Easy wins (free VRAM with no pipeline change)
1. **Close heavy desktop GPU clients** before a session: browser / Edge WebView, Razer App Engine,
   NVIDIA overlay, extra Terminal/Snipping windows. Each returns VRAM to the pipeline.
2. **Disable GPU hardware-acceleration** in any browser/Electron app you keep open.
3. On a **dedicated/headless** box (no interactive desktop) these clients vanish entirely — the
   cleanest fix if this becomes a deployment.

## Allocator hint (already wired)

`scripts/run.ps1` now sets, for the avatar + pipeline processes:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This lets PyTorch grow/shrink its CUDA arena instead of reserving fixed segments, cutting
fragmentation so the same workload fits in **less reserved** VRAM. Pure allocator hint — no behavior change.

## vLLM KV-cache trim (document only — do NOT change the default blindly)

CosyVoice's vLLM reserves a VRAM fraction for weights + KV cache:

- **`COSYVOICE_VLLM_GPU_UTIL`** (default **0.3** ≈ 4.8GB). It must clear vLLM's ~4GB footprint, so the
  **safe floor is ~0.25** — `0.2` (3.26GB) already crashed with "No available memory for the cache
  blocks." Dropping 0.3 → ~0.26 reclaims ~0.5GB *if* the "Available KV cache memory" log line stays
  positive. Test before trusting it.
- **`--max-model-len`** (in `tts/cosyvoice-server/run_vllm_server.sh` / the vLLM args): TTS sequences are
  short, so a smaller `max-model-len` shrinks the KV-cache reservation. Lowering it frees VRAM with no
  quality loss for short TTS prompts.

Combined, these can claw back ~0.5–1.5GB — but both interact with the load-order rule
(**start CosyVoice before MuseTalk**), so change one knob at a time and re-check the vLLM startup log.

## What does NOT work

- **Moving the TTS/avatar to RAM** (CPU-offload / `--cpu-offload-gb` / managed memory): real-time GPU
  inference needs weights/activations in VRAM; offload pages over PCIe every step and destroys the
  latency the system was tuned for. Not viable.
- **A GPU STT model:** the card has no headroom. STT runs on **CPU** instead — `STT_PROVIDER=sherpa`
  (sherpa-onnx streaming, in-process, ~0 VRAM — recommended) or `funasr` (SenseVoice server), which
  sidestep this entirely.
