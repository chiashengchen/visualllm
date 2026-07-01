# FULL-SETUP — VisualLLm end-to-end, from zero (incl. WSL)

A single, self-contained setup guide for the real-time
**speech → STT → LLM → TTS → photoreal talking-head avatar** system. This consolidates
what `INSTALL.md` defers to other files — the **WSL2 + vLLM** TTS build, the **MuseTalk
conda env + weights**, and the **conda SSL gotcha** — so nothing is "see elsewhere".

```
speech → STT → LLM → TTS → lip-sync avatar → audio + video out      (goal: TTFO < 8 s)
```

> Source-of-truth docs: `STATUS.md` (live state), `WORKFLOW.md` §8 (full `.env`),
> `docs/PROBLEMS-AND-FIXES.md` (every bug + fix). `INSTALL.md` is the short version.

---

## 0. The shape of the system (read once)

Four processes cooperate. Ports are fixed by convention:

| # | Process | Where it runs | Port | Started by |
|---|---------|---------------|------|-----------|
| 1 | **CosyVoice TTS** (vLLM) | WSL2 Ubuntu, `cosyvllm` conda env | `:8001` | `run_vllm_server.sh` |
| 2 | **MuseTalk avatar** | Windows, `musetalk` conda env | `:8002` | `scripts/run.ps1` |
| 3 | **Pipeline** (Pipecat, serves `/client`) | Windows, **system Python 3.11** | `:7860` | `scripts/run.ps1` |
| 4 | **Config panel** (optional, edits `.env`) | Windows, system Python | `:7870` | `launch.ps1` |

STT (Deepgram) and the LLM (OpenRouter) are **cloud APIs** by default — no local process.
The GPU is **shared** between CosyVoice (vLLM) and MuseTalk, which is why load order and
VRAM matter (see §4, §10).

> **This box (`E:\Claude\VisualLLm`) is already fully provisioned** — all envs, weights,
> models, and `.env` exist. Jump to **§8 Run**. The build steps below are for a fresh
> machine (or to understand/repair what's here). Per-step "already done here" notes mark
> what you can skip on this box.

---

## 1. Hardware + accounts

| Requirement | Why |
|-------------|-----|
| **1 NVIDIA GPU, ≥16 GB VRAM** | CosyVoice (vLLM) **and** MuseTalk share one card; below ~16 GB they fight for VRAM and the avatar stalls or TTS fails to load. |
| **Windows 11** | Pipeline + MuseTalk + config panel run on Windows. |
| **WSL2 (Ubuntu)** — for the fast TTS path | Only the vLLM TTS needs WSL2. A Windows-only fallback runs without it at ~3× the first-audio latency (§4 Path B). |
| **Miniconda/Anaconda** on Windows **and** in WSL | The avatar/TTS run in dedicated conda envs. |
| **~20 GB free disk** | CosyVoice2-0.5B (~2 GB), MuseTalk + face weights, optional models. |
| **Deepgram key + OpenRouter key** | STT and LLM cloud APIs (both have free tiers). |

Get the two keys (both free-tier):

| Service | Stage | Sign up | Free tier |
|---------|-------|---------|-----------|
| **Deepgram** | STT (`en-US`/`zh-TW`/`th`) | console.deepgram.com | $200 credit |
| **OpenRouter** | LLM (any model) | openrouter.ai | many cheap/free models |

> Without a comparable GPU box this **cannot run end-to-end**. A reviewer can still
> `python -m scripts.preflight` (import check, no GPU) and read the code.

---

## 2. WSL2 + Ubuntu (one-time, Windows)

Needed only for the recommended fast TTS path (§4 Path A). Skip if you'll use Path B.

```powershell
# In an ADMIN PowerShell:
wsl --install -d Ubuntu          # installs WSL2 + Ubuntu; reboot if prompted
wsl --update                     # ensure latest WSL2 kernel (CUDA-in-WSL needs it)
wsl -l -v                        # confirm: Ubuntu ... Running ... VERSION 2
```

- The NVIDIA driver on **Windows** provides CUDA inside WSL2 — do **not** install a
  separate Linux GPU driver. Verify inside WSL: `nvidia-smi` should list your card.
- Install **Miniconda inside Ubuntu** (separate from the Windows one):
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh        # accept defaults -> ~/miniconda3
  ```

> **Already done here:** WSL2 Ubuntu is installed and running; Miniconda lives at
> `~/miniconda3` (WSL user `porsche`). Current WSL IP: `172.24.44.238` (see §4 / §7 — it
> can change on `wsl --shutdown`).

---

## 3. Clone + the pipeline environment (Windows, system Python 3.11)

The pipeline runs on **system Python 3.11** (it has Pipecat — this is **not** a conda env).

```powershell
git clone https://github.com/Triple3Pww/visualllm.git
cd visualllm
pip install -r requirements.txt
python -m scripts.preflight        # verify EVERY import resolves BEFORE wiring keys
```

`preflight` catches Pipecat version drift (its import paths move between releases).
Stages should report `KEYS` (key missing — expected now) or `PASS`, never `ImportError`.

> `requirements.txt` already pulls `sherpa-onnx` + `opencc` (for the optional offline
> STT, §6). **Already done here:** repo cloned at `E:\Claude\VisualLLm`; deps installed.

---

## 4. TTS — CosyVoice2 (pick one path)

Both paths serve the **same** `/tts/stream` raw-PCM contract on port **8001**; the
pipeline reaches it via `.env` `COSYVOICE_URL`. CosyVoice2-0.5B is open-source
(Apache-2.0), strong Mandarin/zh-TW, 24 kHz, zero-shot voice clone.

**Get the upstream code + weights first (both paths need this):**

```bash
# On this box the CosyVoice repo is a SIBLING: E:\Claude\cosyvoice-local-tts
# (the public repo vendors it at tts/cosyvoice-server/ instead — same files).
cd /e/Claude/cosyvoice-local-tts          # or: cd tts/cosyvoice-server  (public layout)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
python -c "from modelscope import snapshot_download; \
  snapshot_download('iic/CosyVoice2-0.5B', \
  local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')"     # ~2 GB
```

(The `CosyVoice/` clone + weights are git-ignored — never committed.)

> **Already done here:** `E:\Claude\cosyvoice-local-tts\CosyVoice\pretrained_models\CosyVoice2-0.5B`
> exists with weights.

### Path A — WSL2 + vLLM (RECOMMENDED, first-chunk TTFB ~1.1 s)

vLLM runs CosyVoice's autoregressive speech-token LLM, cutting first-chunk latency
~3.4 s → ~1.1 s — the original cause of avatar lip-lag.

**One-time: build the `cosyvllm` env inside WSL.** This is a bleeding-edge stack
(vLLM 0.23 + torch 2.11 + cu130, sm_120/Blackwell), and the vendored CosyVoice already
carries the required code patches (model registration, eager mode, env-driven GPU util).
The known-good recipe:

```bash
# inside WSL Ubuntu:
conda create -y -n cosyvllm -c conda-forge python=3.10 c-compiler cxx-compiler ffmpeg
conda activate cosyvllm
pip install vllm==0.23.0                 # pulls torch 2.11.0+cu130 (Blackwell sm_120)
cd /mnt/e/Claude/cosyvoice-local-tts
pip install -r requirements.txt          # CosyVoice deps
pip install lightning matplotlib tensorboard pyarrow torchcodec   # extras the vLLM path needs
# pyworld is SKIPPED on purpose (no wheel, training-only); processor.py imports it optionally.
```

Why each piece (each was a real blocker — all baked into `run_vllm_server.sh`):
- `c-compiler`/`cxx-compiler` (conda-forge, no sudo) — Triton JITs kernels at runtime and
  needs a C/C++ compiler; the script puts the env's `gcc`/`g++` on `CC`/`CXX`+`PATH`.
- `COSYVOICE_VLLM=1` — engine switch to the vLLM LLM (in `tts_engine.py`).
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — run in-process (spawn re-imports `__main__`, crashes).
- `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer's sampler JITs with nvcc (absent) → use torch.
- `COSYVOICE_VLLM_EAGER=1` (default) — skip torch.compile/CUDA-graph capture.
- `COSYVOICE_VLLM_GPU_UTIL=0.3` (default) — vLLM's slice of the **whole** 16 GB card; must
  exceed vLLM's ~4 GB footprint or load crashes "No available memory for the cache blocks".

**Launch it:**
```bash
bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh     # serves :8001 inside WSL
# health (from WSL):  curl 127.0.0.1:8001/health      (~25-35 s to load + warm up)
```

**Then point `.env` at the WSL IP, NOT `localhost`** — WSL2's localhost relay buffers the
stream (~2 s lag):
```bash
wsl hostname -I        # -> e.g. 172.24.44.238 ;  use http://172.24.44.238:8001
```
> The IP can change after `wsl --shutdown` — re-read it and update `COSYVOICE_URL`. (Durable
> alternative: switch WSL to **mirrored networking** so `localhost` works.)

> **Already done here:** the `cosyvllm` env exists at `~/miniconda3/envs/cosyvllm`; weights
> reused from `/mnt/e`. Just run `run_vllm_server.sh`.

### Path B — Windows-only, no WSL2 (fallback, TTFB ~3.4 s)

Run the Windows PyTorch server in a `tts` conda env (Python 3.10; `pynini`+`ffmpeg` from
conda-forge — they don't pip-install):

```powershell
conda create -y -n tts -c conda-forge python=3.10 pynini=2.1.5 ffmpeg
conda activate tts
pip install torch torchaudio
cd E:\Claude\cosyvoice-local-tts        # (or tts\cosyvoice-server in the public layout)
pip install -r requirements.txt
# the conda env's cert store is broken (see §5) -> point SSL at certifi:
$env:SSL_CERT_FILE = (python -c "import certifi;print(certifi.where())")
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```
Then in `.env`: `COSYVOICE_URL=http://localhost:8001`.

> **Drawback:** first-audio TTFB ~3.4 s vs ~1.1 s (≈3× slower), so the avatar's lips
> visibly lag at turn start and the <8 s goal has far less headroom. WSL2 is the *only*
> thing Path A needs that Path B doesn't.

---

## 5. Avatar — MuseTalk (Windows, `musetalk` conda env)

Mouth-region lip-sync on the GPU; serves port **8002**. Runs in **both** TTS paths.

**5.1 Get the upstream clone** under `local_services/musetalk_server/vendor/MuseTalk`
(git-ignored). **5.2 Build the env:**
```powershell
conda create -y -n musetalk python=3.10
conda activate musetalk
cd E:\Claude\VisualLLm\local_services\musetalk_server\vendor\MuseTalk
pip install -r requirements.txt
pip install --editable ./musetalk/whisper          # whisper audio encoder
pip install --no-cache-dir -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
```

**5.3 Download the model weights** into `vendor/MuseTalk/models/` (MuseTalk model,
sd-vae-ft-mse, whisper tiny, dwpose, face-parse-bisent). MuseTalk ships a
`download_weights.bat`/`.sh`; or fetch manually per its `models/README.md`. Layout:
```
models/  musetalk/  musetalkV15/  sd-vae/  whisper/  dwpose/  face-parse-bisent/
```

**5.4 The CONDA SSL GOTCHA (will bite you at first start).** The `musetalk`/`tts` conda
envs have a **broken Windows cert store** (`ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`) that
kills `torch.hub`/`urllib` HTTPS downloads. On a fresh `AVATAR_REF`, MuseTalk one-time
downloads the **face-detector** weights and crashes there. Fix (both halves):
```powershell
# (a) Pre-fetch the two face weights with curl (system SSL) into torch's cache:
curl -L -o "$env:USERPROFILE\.cache\torch\hub\checkpoints\s3fd-619a316812.pth" ^
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
curl -L -o "$env:USERPROFILE\.cache\torch\hub\checkpoints\2DFAN4-11f355bf06.pth.tar" ^
  https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar
# (b) For any pip/modelscope (requests-based) download, set both to certifi's bundle:
$env:SSL_CERT_FILE     = (python -c "import certifi;print(certifi.where())")
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
```
Once cached, `load_state_dict_from_url` skips the network. If a torch conda env on this
box "fails to download" at startup, it's **this**, not a network outage.

**5.5 Portrait / size / fps** come from `.env`: `AVATAR_REF` (default
`assets/avatar_female.png`), `MUSETALK_SIZE`, `MUSETALK_FPS`. The server reads these from
the **OS env only** (no python-dotenv in its env) — `scripts/run.ps1` propagates them.

> **Already done here:** `musetalk` env exists; `vendor/MuseTalk/models/` is populated;
> the two face weights are cached in `C:\Users\MARU\.cache\torch\hub\checkpoints\`.

---

## 6. (Optional) Offline STT + MOSS TTS — fully local

Default STT is **Deepgram** (cloud). For a fully-offline STT (CPU, ~0 VRAM, no GPU
contention):

**Option A — `sherpa` (streaming, in-process — RECOMMENDED).** sherpa-onnx zipformer,
bilingual zh-en, runs **in the pipeline** (system Python, no server). Drives turn-taking
from its **own ASR endpoint detector**, so a quiet/remote mic still flushes the turn (the
failure mode that breaks segmented STT). Deps are already in `requirements.txt`; download
the model (~330 MB) into `models/` (git-ignored):
```bash
curl -L -o models/m.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
cd models && tar xjf m.tar.bz2 && cd ..
```
Then `.env`: `STT_PROVIDER=sherpa` (knobs `SHERPA_ENDPOINT_SILENCE=0.5`, `SHERPA_TRADITIONAL=1`).
> **Already done here:** the sherpa model is unpacked under `models/`.

**Option B — `funasr` (segmented SenseVoice server, `:8004`).**
```powershell
conda create -y -n funasr-stt python=3.11
conda activate funasr-stt
pip install -r local_services/funasr_server/requirements.txt   # model ~1GB auto-downloads
```
`.env`: `STT_PROVIDER=funasr`; `run.ps1` auto-starts `:8004`. **Caveat:** segmented — needs
the energy-VAD to fire end-of-turn; on a too-quiet mic the turn never flushes. Prefer
`sherpa`. (Env exists here.)

**Optional MOSS-TTS** (`TTS_PROVIDER=moss`, `:8003`, `moss-tts` conda env) — alternative
streaming TTS, same PCM contract. Launch recipe (Triton/torchcodec/ffmpeg-7 fixes) is in
`local_services/moss_server/app.py`'s docstring. Run it **eager** (default).

---

## 7. Configure `.env`

```powershell
copy .env.example .env        # then edit
```
Minimum to fill:
```ini
LANGUAGE=en                              # en | zh | th
DEEPGRAM_API_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemini-2.5-flash-lite   # cloud (accurate default). Local Ollama: base_url->:11434/v1 + model. For a reasoning model (qwen3.5:4b) add OPENROUTER_REASONING_EFFORT=none (else ~33s thinking) + OPENROUTER_MAX_TOKENS to cap length. See WORKFLOW.md §8.

STT_PROVIDER=deepgram                    # or sherpa (offline) / funasr
TTS_PROVIDER=cosyvoice
COSYVOICE_URL=http://172.24.44.238:8001  # Path A: the WSL IP (wsl hostname -I). Path B: http://localhost:8001
MUSETALK_SYNC_MODE=steady                # steady = video-master, synced start (default) | live = voice-instant, lips trail
MUSETALK_TRT=1                           # TensorRT render (default). Build engines first (below) or it falls back to PyTorch
```
Full knob reference: `WORKFLOW.md` §8. **Reminder:** after `wsl --shutdown` the WSL IP can
change — re-check `wsl hostname -I` and update `COSYVOICE_URL`.

**TensorRT engines (`MUSETALK_TRT=1`, the baseline).** The avatar defaults to the TensorRT render path
(~1.5× faster; holds A/V sync under shared-GPU contention where PyTorch drifts on long turns —
`docs/PROBLEMS-AND-FIXES.md` P16). The engines are **not** shipped (GPU/driver-specific, ~1.75 GB,
gitignored in `local_services/musetalk_server/trt_cache/`). If they are absent the server logs it and
**falls back to the PyTorch path** (correct, just slower), so this is optional-but-recommended. Build them
once in the `musetalk` env (there is no CLI — the export/build are helper functions; run these four
one-liners, ~7 min; rebuild after a GPU/driver change):
```powershell
$py = "E:\miniconda3\envs\musetalk\python.exe"
# 0) install the TRT libs (from NVIDIA's index)
& $py -m pip install -r local_services/musetalk_server/requirements-trt.txt --extra-index-url https://pypi.nvidia.com
# 1) UNet: torch -> ONNX -> fp16 engine
& $py -c "from local_services.musetalk_server.app import engine; engine.load(); from local_services.musetalk_server.trt_export import export_unet_onnx; export_unet_onnx(engine,'local_services/musetalk_server/trt_cache/unet.onnx')"
& $py -c "from local_services.musetalk_server.trt_build import build_engine; S=50; build_engine('local_services/musetalk_server/trt_cache/unet.onnx','local_services/musetalk_server/trt_cache/unet.engine',{'latent':((1,8,32,32),(8,8,32,32),(8,8,32,32)),'audio':((1,S,384),(8,S,384),(8,S,384)),'timestep':((1,),(1,),(1,))})"
# 2) VAE decoder: torch -> ONNX -> fp16 engine
& $py -c "from local_services.musetalk_server.app import engine; engine.load(); from local_services.musetalk_server.trt_export import export_vae_onnx; export_vae_onnx(engine,'local_services/musetalk_server/trt_cache/vae.onnx')"
& $py -c "from local_services.musetalk_server.trt_build import build_engine; build_engine('local_services/musetalk_server/trt_cache/vae.onnx','local_services/musetalk_server/trt_cache/vae.engine',{'latent':((1,4,32,32),(8,4,32,32),(8,4,32,32))})"
```

---

## 8. Run

**Easiest — one click:** double-click **`Run VisualLLm.exe`** in the repo root. It brings
up WSL CosyVoice (waits on `/health`), then avatar+pipeline, then the config panel, then
opens the client. The launcher window is the on/off switch — press **Enter** (or close it)
to stop everything.

**Manual — the load order MATTERS (start CosyVoice BEFORE MuseTalk; §10):**
```powershell
# 1. CosyVoice TTS (Path A: in WSL)
wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"
# 2 + 3. MuseTalk avatar + pipeline (one script; propagates MuseTalk env from .env)
.\scripts\run.ps1
# 4. (optional) config panel — edit .env + restart the pipeline from the browser
python -m local_services.config_panel.server          # http://localhost:7870
```

Then open **`http://localhost:7860/client/`** — **WITH the trailing slash** (without it the
prebuilt page 404s its assets → white screen). Allow the mic, **wait for the avatar face to
appear**, then talk.

---

## 9. Verify it works

- `python -m scripts.preflight` — every import resolves; stages flip `KEYS` → `PASS` once keys set.
- Each turn logs a **`[TTFO]`** line (time-to-first-output). **Pass = p95 < 8 s** (printed on disconnect).
- Headless A/V timeline + metrics (no browser): `python -m scripts.measure --offline-capture`.

---

## 10. Troubleshooting (the load-bearing gotchas)

1. **Avatar won't talk / "Cannot connect to host …:8001":** CosyVoice isn't up or crashed
   for VRAM. **Start CosyVoice (vLLM) BEFORE MuseTalk** — at `gpu_util 0.3` vLLM needs the
   card mostly free; starting it while MuseTalk holds ~5 GB crashes "No available memory for
   the cache blocks". Clean recovery: stop all three → start cosyvoice on the near-empty
   card → then `run.ps1`. (The `.exe` launcher already does this order.)
2. **VRAM looks full (`nvidia-smi` ~16 GB used):** much of that is **Windows desktop apps**
   (explorer, EdgeWebView2, NVIDIA overlay, Razer…), not the pipeline — closing them frees
   VRAM for free. `run.ps1` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   See `docs/gpu-memory-notes.md`.
3. **Audio lags ~2 s on Path A:** `COSYVOICE_URL` is `localhost`, not the WSL IP.
4. **White screen at the client:** you opened `/client` without the trailing slash. Use `/client/`.
5. **Conda env "fails to download" weights at startup:** the broken cert store — §5.4.
6. **Chinese voice starts ~1 s later than English:** known, unfixable on one GPU
   (CosyVoice's zh first-chunk TTFB; `COSYVOICE_FIRST_HOP` fixes it but starves the avatar).
   Don't enable it. `docs/PROBLEMS-AND-FIXES.md` P15.
7. **Avatar choppy/desynced over RDP/WAN:** RDP adds its own choppiness — judge sync with
   the offline capture tools, not the remote window. Remote viewing is best over Tailscale
   (`tailscale serve` HTTPS) in a native browser.
8. **Anything else:** read **`docs/PROBLEMS-AND-FIXES.md`** — nearly every failure here is
   already diagnosed and fixed there.
