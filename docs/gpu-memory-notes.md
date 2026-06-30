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

> ⚠️ **Superseded by the measured breakdown below (same day).** This section was *inferred*
> from the GPU **process list** (seeing desktop apps listed) — NOT measured. A stop-and-diff
> measurement later that day proved the desktop holds only **~0.9 GB**, not the bulk. Closing
> desktop apps reclaims almost nothing here. Keep reading.

A snapshot showed **15796 / 16311 MiB used, 255 MiB free** — and the GPU process list was full of
**non-pipeline apps**: `explorer.exe`, `SearchHost`, `StartMenuExperienceHost`, `msedgewebview2.exe`,
`WindowsTerminal`, `NVIDIA Overlay`, `RazerAppEngine`, `SnippingTool`, `TextInputHost`, `ShellExperienceHost`.
Seeing them *listed* suggested they were the cause — but a process being a GPU **client** (it can draw)
is not the same as it **holding GBs**. The measurement below corrects the conclusion.

## Measured breakdown (2026-06-30, stop-and-diff — the authoritative numbers)

Because per-process `used_memory` is `[N/A]` on this WDDM consumer card (both the Windows **and** WSL
`nvidia-smi` show N/A — a driver limitation, not permissions), the only reliable attribution is to
**stop each piece and diff `memory.used`**. Doing that:

| State | `memory.used` | Attributed to |
|-------|---------------|---------------|
| Full stack running | ~15,554–15,716 MiB | everything |
| **Stack fully down** (pipeline + MuseTalk + WSL vLLM killed) | **942 MiB** | **Windows desktop = ~0.9 GB total** |
| vLLM loaded, MuseTalk/pipeline down | 6,970 MiB | **CosyVoice vLLM ≈ ~6 GB** |
| + MuseTalk + pipeline (fresh) | 15,716 MiB | **MuseTalk ≈ ~8.7 GB**; pipeline ≈ **0 GB** (CPU-only — not even in the GPU process list) |

**Conclusions (these correct the earlier section):**
1. **The desktop is ~0.9 GB, not the bulk.** Closing apps (Photos, Snipping Tool, Edge GameAssist,
   VS Code, even all of Edge's 29 procs) freed **~168 MiB total** — noise. Desktop cleanup is a dead end here.
2. **Our stack genuinely needs ~14.7 GB** (vLLM ~6 + MuseTalk ~8.7). The card is full because the two
   models legitimately fill it.
3. **A clean restart reclaims nothing** — 15,554 → 15,716 MiB after stopping and reloading everything
   fresh. There is no fragmentation/leak to claw back; the stack uses ~14.7 GB cold.
4. **The pipeline process uses ~0 GPU** — it's CPU-only (WebRTC + audio resampling).

### The lag is COMPUTE, not memory (the load-bearing takeaway)
With the stack needing ~14.7 GB and only ~0.6 GB free, the avatar still lagged ~4 s — **but GPU
*compute* sat at ~1% between turns.** So the ~4 s lip-trail is **compute contention** (vLLM generating
speech tokens while MuseTalk renders, on the same SMs during a turn), **not** VRAM pressure. Measured
that session: LLM TTFB ~0.6–0.9 s, CosyVoice TTFB ~1.2 s (both fine); avatar lips start **+4.1–4.6 s
after voice** and hold ~4 s behind at ~12 fps. **Freeing VRAM does not touch this.** The levers that do:
`MUSETALK_SYNC_MODE=live` (voice instant, lips trail — no VRAM involved) or a **dedicated avatar GPU**.

### Easy wins — small, but real (free VRAM with no pipeline change)
1. **Disable GPU hardware-acceleration** in any browser/Electron app you keep open — durable, unlike
   force-closing (which the apps undo on reopen).
2. On a **dedicated/headless** box (no interactive desktop) the ~0.9 GB of desktop clients vanish — but
   that's <1 GB, and won't fix the compute-bound lag.
3. (Closing desktop apps mid-session: **don't bother** — measured ~168 MiB, not worth the disruption.)

## ⭐ VRAM trim — measured before/after (2026-06-30, branch feat/offline-stt-sensevoice)

A full stop-and-diff pass turned the stack from "card is full" into "fits a 10GB card with room"
(and an 8GB card with one extra knob). **~15.7GB → ~8.4GB working — ~7.3GB reclaimed**, far beyond
the ~2–3GB first estimated. All levers are reversible env/arg knobs; defaults preserved.

| Lever | State measured | Before | After | Reclaimed |
|-------|----------------|--------|-------|-----------|
| **vLLM `gpu_util` 0.30→0.16 + `max_model_len`=2048** | vLLM-only working | 6029 MiB | **3735 MiB** | **−2294** |
| **MuseTalk `empty_cache()` after warmup** | MuseTalk working (mid-turn peak, stable across turns) | 8761 MiB | **~4844 MiB** | **−3917** |
| MuseTalk `MUSETALK_BATCH` 8→4 (optional knob) | mid-turn peak | 9369 MiB | 8309 MiB | −1060 |
| **Full stack** (vLLM 0.16 + empty_cache, batch 8) | mid-turn | **~15676 MiB** | **~8400 MiB** | **~7300** |

**The big realization (why this beat the old ~0.5–1.5GB estimate):** CosyVoice generates **one short
sentence** of speech tokens per request, but vLLM was sizing its KV cache for the model's default
`max_model_len` of **32,768 tokens** — a massive, almost-entirely-wasted reservation. Capping
`max_model_len` to 2048 (`COSYVOICE_VLLM_MAX_LEN`, new) lets the util fraction drop **far below** the
old "~0.25 floor" (that floor only existed *because* of the uncapped KV need). Measured KV headroom:
0.16 → 1.74 GiB / **74× concurrency**; 0.12 → 1.11 GiB / **47×** — both wildly more than one user needs.
And MuseTalk's "8.7GB" was mostly **load-time transients** (fp32 weight copies before `.half()`, warmup
cuDNN workspaces) left in PyTorch's caching allocator — a single `empty_cache()` after warmup returns
them; the real working set is ~4.8GB (verified stable across repeated turns, doesn't climb).

### The knobs (all verified this session)
- **`COSYVOICE_VLLM_GPU_UTIL`** — now defaults **0.16** in `run_vllm_server.sh` (robust, 74× KV margin).
  Set **0.12** to squeeze the whole stack **under 8GB** (47× margin — verified: en + a 27s zh paragraph
  synth clean, no truncation; KV log positive). The crash floor is where `util×16GB − ~830MiB weights`
  goes near-zero; 0.12 (~1.96GB pool) is safe. **Old "0.25 floor" is OBSOLETE** — it assumed uncapped KV.
- **`COSYVOICE_VLLM_MAX_LEN`** — now defaults **2048** in `run_vllm_server.sh`. The lever that makes the
  low util safe. Raise it only if you ever feed unusually long single prompts (TTS never does).
- **`MUSETALK_BATCH`** — keep **8** (the smooth default). Set **4** to shave ~1GB more peak for a tighter
  card; renders fine uncontended (~45ms/frame at 12fps), but smaller batches add per-frame launch
  overhead, so re-check the avatar feel under load before trusting it on the shared GPU.
- Lower util is also **friendlier to the load order** — vLLM reserving less of the shared card means
  less "No available memory for the cache blocks" when MuseTalk loads after it.

> **Lag is a separate axis (unchanged by all of the above).** The ~4s lip-trail was always COMPUTE
> contention, not VRAM (see below) — freeing 7GB does not touch it. The current `steady` default feels
> good on `/client`; the bounded-`live` alternative (`MUSETALK_OUT_Q`, below) and a TensorRT'd MuseTalk
> remain the genuine lag levers. **Newly viable:** the avatar alone now fits ~4.8GB, so a cheap
> dedicated avatar GPU (which *also* kills the contention) is an option it wasn't before.
>
> **`MUSETALK_OUT_Q`** (new, default 600 ≈ unbounded) bounds the avatar server's rendered-frame queue.
> In `live` mode a tight value (tested **24** ≈ 1.2s @20fps) caps how far the lips can trail the voice
> by dropping stale frames instead of accumulating a backlog — verified smooth (12fps, no freeze) on the
> WebRTC delivery path. Do **not** re-lock the voice to video (that froze it).

## ⭐ MuseTalk TensorRT render path — BUILT + VALIDATED (2026-06-30, `MUSETALK_TRT=1`)

The compute-contention lever (spec #2) is implemented and measured. The UNet + VAE-decoder run as
fp16 TensorRT engines instead of PyTorch; whisper/PE/compositing stay torch. `MUSETALK_TRT=1` switches
it on (default **0** = the proven torch path; any engine load failure falls back to torch).

| Metric | PyTorch | TensorRT | Result |
|--------|---------|----------|--------|
| render_segment end-to-end | 43.2 ms/frame | 23.6 ms/frame | **1.83×** |
| GPU compute only (ex-composite) | ~33 ms/frame | ~13.5 ms/frame | **~2.4×** |
| Frame quality (SSIM vs torch, composited) | — | — | **1.0000 (identical)** |
| Contention (full stack, vLLM streaming) | baseline | 12.0 fps, no freeze | keeps up |

- **Headroom is the point:** at 12 fps (83 ms budget) the render now uses ~28% of budget (was ~52%) —
  spare capacity for higher fps/quality, a near-zero-trail `live`, or freeing SMs for zh TTS.
- **VRAM: TRT costs ~+0.5 GB, not less** (full-stack peak ~9.95 GB vs ~9.4 GB torch). With `MUSETALK_TRT=1`
  the torch UNet/VAE stay resident as the runtime fallback **plus** the ~1.75 GB engines. TRT's win is
  latency, not footprint (as the spec predicted). **Follow-up to reclaim it:** free the torch UNet/VAE
  after engines load (drops the fallback) → the TRT path would then be *smaller* than torch.
- **Build:** engines are a prebuilt artifact (`trt_cache/{unet,vae}.engine`, ~1.75 GB, ~7 min to build
  via `trt_build.py`; gitignored). Needs `tensorrt-cu12` (`--extra-index-url https://pypi.nvidia.com`)
  + `onnx` in the `musetalk` env. Blackwell `sm_120` confirmed. Full recipe:
  `docs/superpowers/plans/2026-06-30-musetalk-tensorrt.md`.
- **To actually USE the headroom:** enable `MUSETALK_TRT=1` and raise `MUSETALK_FPS` (16/20 now affordable)
  — the render has the budget for it. Default stays torch+12fps (the proven smooth baseline).

## What does NOT work / was DISPROVEN

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a NO-OP on Windows.** `run.ps1` sets it, but
  the MuseTalk server logs `UserWarning: expandable_segments not supported on this platform` — the CUDA
  caching allocator's expandable-segments mode is Linux-only. It does no harm, but it does **not** cut
  fragmentation here (the earlier note claiming it helps was wrong). The `empty_cache()`-after-warmup
  trim above is what actually reclaims the reserved blocks on Windows.

## What does NOT work

- **Moving the TTS/avatar to RAM** (CPU-offload / `--cpu-offload-gb` / managed memory): real-time GPU
  inference needs weights/activations in VRAM; offload pages over PCIe every step and destroys the
  latency the system was tuned for. Not viable.
- **A GPU STT model:** the card has no headroom. STT runs on **CPU** instead — `STT_PROVIDER=sherpa`
  (sherpa-onnx streaming, in-process, ~0 VRAM — recommended) or `funasr` (SenseVoice server), which
  sidestep this entirely.
