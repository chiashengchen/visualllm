# INSTALL — VisualLLm from zero

A real-time **speech → STT → LLM → TTS → photoreal talking-head avatar** system.
You speak; a lip-synced avatar speaks back. Multi-turn, streaming end-to-end, with a
goal of **time-to-first-output < 8 s**.

```
speech → STT → LLM → TTS → lip-sync avatar → audio + video out
```

This guide takes a fresh machine to a running system. For the *why* behind each
choice and the live state, see `README.md`, `STATUS.md`, `WORKFLOW.md` (full `.env`
reference in §8), and `docs/PROBLEMS-AND-FIXES.md` (every bug + fix — read before
re-debugging the avatar/audio).

---

## ⚠️ Hardware reality check (read first)

This is a **local, GPU-bound** system. You need:

| Requirement | Why |
|-------------|-----|
| **1 NVIDIA GPU, ≥16 GB VRAM** | The TTS engine (CosyVoice on vLLM) **and** the MuseTalk avatar renderer share **one** card. Below ~16 GB they fight for VRAM and the avatar stalls or the TTS fails to load. |
| **Windows 11** | The pipeline + MuseTalk + config panel run on Windows. |
| **WSL2 (Ubuntu)** — *recommended, optional* | Only the **fast** TTS path (vLLM) needs WSL2. A Windows-only fallback runs without it at a latency cost (see [TTS](#4-tts-pick-one-of-two-paths)). |
| **~20 GB free disk** | Model weights (CosyVoice2-0.5B ~2 GB, MuseTalk + face-detection weights, optional MOSS). |
| **A Deepgram key + an OpenRouter key** | STT and LLM are cloud APIs (both have free tiers). |

**Without a comparable GPU box this cannot run end-to-end.** A reviewer can still read
the code, run `python -m scripts.preflight` (import check, no GPU), and inspect the
benchmark harness output — but live conversation needs the hardware above.

---

## 1. Get the API keys

| Service | Stage | Sign up | Free tier |
|---------|-------|---------|-----------|
| **Deepgram** | STT (nova-2; `en-US`/`zh-TW`/`th`) | console.deepgram.com | $200 credit |
| **OpenRouter** | LLM (any model via `OPENROUTER_MODEL`) | openrouter.ai | pay-as-you-go; many cheap/free models |

Optional cloud TTS fallbacks (not needed for the default local stack): ElevenLabs,
Deepgram Aura. See `docs/SETUP.md` for those.

---

## 2. Clone the repo

```bash
git clone https://github.com/Triple3Pww/visualllm.git
cd visualllm
```

---

## 3. Pipeline environment (Windows, system Python 3.11)

The pipeline runs on **system Python 3.11** (it has Pipecat — this is *not* a conda env).

```bash
pip install -r requirements.txt
python -m scripts.preflight          # verify every import resolves BEFORE wiring keys
```

`preflight` catches Pipecat version drift (its import paths shift between releases).
Selected stages should report `KEYS` (key missing, expected now) or `PASS`, never an
`ImportError`.

---

## 4. TTS — pick one of two paths

CosyVoice2-0.5B is vendored in this repo at **`tts/cosyvoice-server/`** (wrapper code
only). Both paths serve the same `/tts/stream` raw-PCM contract on port **8001**; the
pipeline reaches it via `COSYVOICE_URL`.

First, get the upstream CosyVoice code + weights into the vendored folder (both paths
need this):

```bash
cd tts/cosyvoice-server
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
python -c "from modelscope import snapshot_download; \
  snapshot_download('iic/CosyVoice2-0.5B', \
  local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')"     # ~2 GB
cd ../..
```

(The `CosyVoice/` clone and weights are git-ignored — never committed.)

### Path A — WSL2 + vLLM (recommended, first-chunk TTFB ~1.1 s)

vLLM runs CosyVoice's autoregressive speech-token LLM, cutting first-chunk latency
~3.4 s → ~1.1 s — the original cause of avatar lip-lag.

```bash
# inside WSL2 Ubuntu — one-time: a `cosyvllm` conda env with vLLM + a conda gcc/g++
#   (Triton JITs kernels at runtime and needs the compiler; flashinfer is disabled).
# Then launch (the script self-locates the repo and the env):
bash tts/cosyvoice-server/run_vllm_server.sh         # serves :8001 inside WSL
```

Then set, in `.env`, **`COSYVOICE_URL` to the WSL IP, NOT `localhost`** — WSL2's
localhost relay buffers the streaming audio (~2 s lag):

```bash
wsl hostname -I        # -> e.g. 172.x.x.x ; use http://172.x.x.x:8001
```

VRAM note: vLLM's `gpu_memory_utilization` (env `COSYVOICE_VLLM_GPU_UTIL`, default
`0.3`) must clear vLLM's ~4 GB footprint. **Start CosyVoice BEFORE MuseTalk** — at
util 0.3 vLLM needs the card mostly free, or it crashes "No available memory for the
cache blocks." (`docs/PROBLEMS-AND-FIXES.md` P15.)

### Path B — Windows-only, no WSL2 (fallback, first-chunk TTFB ~3.4 s)

Run the Windows PyTorch server in a `tts` conda env (Python 3.10; `pynini` + `ffmpeg`
from conda-forge — see `tts/cosyvoice-server/README.md`):

```bash
# from tts/cosyvoice-server, in the `tts` conda env:
set SSL_CERT_FILE=<path to certifi cacert.pem>     # conda env's cert store is broken
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

Then in `.env`: `COSYVOICE_URL=http://localhost:8001`.

> **Drawback (the reason Path A exists):** first-chunk TTFB is ~**3.4 s** vs ~1.1 s —
> roughly **3× slower to first audio**, so the avatar's lips visibly lag at the start
> of each turn and the < 8 s TTFO goal has far less headroom. WSL2 is the *only* thing
> Path A needs that Path B doesn't; everything else (pipeline, avatar) is identical.

---

## 5. Avatar — MuseTalk (Windows, `musetalk` conda env)

MuseTalk does mouth-region lip-sync on the GPU and serves port **8002**. It runs on
Windows in **both** TTS paths.

1. Create a `musetalk` conda env and install MuseTalk's deps (the server is
   `local_services/musetalk_server/app.py`; the upstream clone + weights live under
   `local_services/musetalk_server/vendor/`, git-ignored).
2. Download the weights — **s3fd** + **2DFAN4** (face detect/align) and the **MuseTalk**
   model. The `musetalk`/`tts` conda envs have a **broken Windows cert store** that kills
   `torch.hub`/`urllib` downloads, so curl-cache the weights and set `SSL_CERT_FILE` to
   certifi (`docs/PROBLEMS-AND-FIXES.md` and `STATUS.md` cover the exact recipe).
3. The portrait is set by `AVATAR_REF`; size/fps by `MUSETALK_SIZE` / `MUSETALK_FPS`.

`scripts/run.ps1` starts the avatar server **and** the pipeline together, propagating
the MuseTalk knobs from `.env`.

---

## 6. (Optional) MOSS-TTS-Realtime

An alternative streaming TTS (`local_services/moss_server/app.py`, `moss-tts` conda
env, port **8003**). Speaks the same PCM contract as CosyVoice. Enable with
`TTS_PROVIDER=moss`. Launch recipe (Triton/torchcodec/ffmpeg-7 fixes) is in the
server's module docstring. Run it **eager** (the default) to avoid recompile stalls.

---

## 7. Configure `.env`

```bash
copy .env.example .env        # Windows (cp on WSL/Unix)
```

Fill in:

```ini
LANGUAGE=en                   # en | zh | th
DEEPGRAM_API_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini      # any OpenRouter model

TTS_PROVIDER=cosyvoice        # cosyvoice (default) | moss | elevenlabs | deepgram
COSYVOICE_URL=http://<WSL-IP>:8001       # Path A: WSL IP. Path B: http://localhost:8001

MUSETALK_SYNC_MODE=steady     # steady = video-master, synced start (default) | live = voice-instant, lips trail
```

Full knob reference: `WORKFLOW.md` §8.

---

## 8. Run

**Easiest — one click:** double-click **`Run VisualLLm.exe`** in the repo root. It
brings up the WSL CosyVoice TTS (waits on `/health`), then the avatar + pipeline, then
the config panel, and opens the client. The launcher window is the on/off switch —
press Enter (or close it) to stop everything.

**Manual — 3 processes:**

```bash
# 1. CosyVoice TTS  (Path A: in WSL)
bash tts/cosyvoice-server/run_vllm_server.sh
# 2 + 3. MuseTalk avatar server + pipeline (one script; propagates MuseTalk env from .env)
.\scripts\run.ps1
```

**Web config panel** (edit `.env` + restart the pipeline from a browser):

```bash
python -m local_services.config_panel.server        # http://localhost:7870
```

Then open **`http://localhost:7860/client/`** — **with the trailing slash** (without
it the prebuilt page 404s its assets → white screen). Allow the mic, **wait for the
avatar face to appear**, then talk.

---

## 9. Verify it works

- `python -m scripts.preflight` — every import resolves; selected stages flip `KEYS` → `PASS` once keys are set.
- The console logs a **`[TTFO]`** line per turn (time-to-first-output). **Pass = p95 < 8 s** (printed on disconnect).
- Avatar A/V timeline + metrics, headless (no browser): `python -m scripts.measure --offline-capture`.

---

## 10. Troubleshooting (top gotchas)

1. **Avatar won't talk / "Cannot connect to host …:8001":** CosyVoice isn't up, or it
   crashed for VRAM. **Start CosyVoice before MuseTalk**; free VRAM if loading fails.
2. **Audio lags ~2 s on Path A:** `COSYVOICE_URL` is `localhost`, not the WSL IP. Use `wsl hostname -I`.
3. **White screen at the client:** you opened `/client` without the trailing slash. Use `/client/`.
4. **Avatar choppy/desynced when viewed remotely (RDP/WAN):** RDP adds its own
   choppiness — judge sync with the offline capture tools, not the remote window.
   Remote viewing is best over Tailscale (`tailscale serve` HTTPS) in a native browser.
5. **Anything else** — read **`docs/PROBLEMS-AND-FIXES.md`**; nearly every failure mode
   here has already been diagnosed and fixed there.
