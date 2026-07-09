# Local TTS — CosyVoice2 (zh-TW / English)

Local, open-source Text-to-Speech for the **AI Weather Forecaster Virtual Human**
project (NCU). Replaces ElevenLabs in the `STT → LLM → TTS → MuseTalk` pipeline
with a fully local **CosyVoice2-0.5B** engine, exposed as a FastAPI service on
port **8001** (matches the pipeline's `COSYVOICE_URL`).

> **Chinese-quality fixes — apply `patches/` after cloning upstream CosyVoice.** Running the LLM on
> vLLM dropped CosyVoice's repetition-aware sampling, so Chinese intermittently looped on silence
> (a ~4 s sentence → ~12 s of dead air). `patches/` restores it (a vLLM logits processor + `top_p`),
> and `tts_engine.py` now defaults to a naturally fluid "pro" voice so zh ≈ English pacing. See
> `patches/README.md` and `docs/PROBLEMS-AND-FIXES.md` P18.

```
tts/
├── app.py             # FastAPI service: web page + POST /tts + /health
├── tts_engine.py      # shared CosyVoice2 wrapper (load once, synthesize)
├── templates/
│   └── index.html     # demo web UI (text box, Generate, audio player)
├── static/
│   ├── style.css      # web UI styling (responsive)
│   └── app.js         # web UI logic (calls POST /tts)
├── test_en.py         # English MWE  -> outputs/output_en.wav
├── test_zh.py         # Mandarin MWE -> outputs/output_zh.wav
├── benchmark.py       # latency / RTF
├── client_example.py  # call the service from another machine
├── requirements.txt   # pip deps (pynini/ffmpeg come from conda — see below)
├── .gitignore
├── README.md
├── outputs/           # generated wav files (git-ignored)
└── CosyVoice/         # upstream repo + model (git-ignored; clone separately)
```

## Why CosyVoice2

Local, Apache-2.0/open-source, strong Mandarin (incl. zh-TW prosody), 24 kHz,
designed for digital-human systems, supports zero-shot voice cloning and
fine-grained control (`[laughter]`, `<strong>`). Good research value.

---

## Installation (Apple Silicon, macOS)

CosyVoice targets **Python 3.10**. Two dependencies (`pynini`, `ffmpeg`) do **not**
pip-install on macOS arm64 — install them from **conda-forge** first.

```bash
# 1. Python 3.10 env with the conda-only deps
conda create -y -n tts -c conda-forge python=3.10 pynini=2.1.5 ffmpeg
conda activate tts

# 2. PyTorch (arm64 wheel ships with MPS) + the rest
pip install torch torchaudio
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
pip install -r requirements.txt          # curated, Mac-safe subset

# 3. Download the model (~2 GB) into CosyVoice/pretrained_models/
python -c "from modelscope import snapshot_download; \
  snapshot_download('iic/CosyVoice2-0.5B', \
  local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')"
```

> The `openai-whisper` pin in upstream's requirements fails to build on modern
> setuptools (`No module named 'pkg_resources'`); install the **current**
> `openai-whisper` instead — it provides the `log_mel_spectrogram` /
> `tokenizer` functions CosyVoice2 needs at inference. Already handled in
> `requirements.txt`.

---

## Usage

```bash
conda activate tts
cd tts

python test_en.py        # -> outputs/output_en.wav
python test_zh.py        # -> outputs/output_zh.wav
python benchmark.py      # latency / RTF table

# API + web demo
python -m uvicorn app:app --host 0.0.0.0 --port 8001
curl -X POST http://localhost:8001/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天台北天氣晴朗，氣溫二十八度。"}' --output out.wav
```

### Web demo interface
With the server running, open **http://localhost:8001** in a browser. Type
English or Traditional Chinese, click **Generate Speech**, and the audio plays
in the page. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Demo web page (text box, Generate button, audio player) |
| POST | `/tts` | `{"text": "...", "speed": 1.0}` → `audio/wav` (used by the avatar pipeline **and** the web page) |
| GET | `/health` | `{"status":"ok","device":...,"sample_rate":24000}` |
| GET | `/info` | Service metadata |

The web page calls the same `POST /tts` endpoint — the engine and API are
unchanged, so the `LLM → TTS → MuseTalk` integration is unaffected.

### Changing the voice
Edit `PROMPT_WAV` / `PROMPT_TEXT` in `tts_engine.py` to point at a different
reference clip (3–10 s, 16 kHz) and its transcript. Everything else is unchanged.

---

## Phase 6 — Calling from another machine (LAN)

1. Run the service bound to all interfaces: `uvicorn app:app --host 0.0.0.0 --port 8001`.
2. Find the host IP: `ipconfig getifaddr en0` (macOS) → e.g. `192.168.1.50`.
3. Ensure both machines are on the same LAN; allow inbound 8001 in the firewall.

**curl (from the Windows / MuseTalk box):**
```bash
curl -X POST http://192.168.1.50:8001/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天台北天氣晴朗"}' --output reply.wav
```

**Python client:**
```bash
python client_example.py "今天台北天氣晴朗" --host 192.168.1.50 --out reply.wav
```

Response headers `X-Generation-Seconds`, `X-Audio-Seconds`, `X-RTF` expose
timing for monitoring.

---

## Phase 7 — Final Report

### 1. Installation steps
See *Installation* above. Net path: conda 3.10 env → `pynini`+`ffmpeg` from
conda-forge → `torch` → curated `requirements.txt` → ModelScope model download.

### 2. Issues encountered & fixes
| Issue | Cause | Fix |
|---|---|---|
| Default Python 3.13 (Anaconda) | CosyVoice needs 3.10; no 3.13 ML wheels | Dedicated `conda create python=3.10` env |
| `pynini` won't pip-install on arm64 | needs OpenFst compile | `conda install -c conda-forge pynini=2.1.5` |
| `openai-whisper==20231117` build fails | legacy `setup.py` imports `pkg_resources`; new setuptools dropped it | install current `openai-whisper` |
| ffmpeg missing | not preinstalled | conda-forge `ffmpeg` |
| No CUDA on Mac | Apple Silicon | runs on CPU (jit/trt/fp16 auto-disabled) |

### 3. Performance results
_Filled in from `benchmark.py` on the M4 (CPU) — see results section appended
after the benchmark run._

### 4. Chinese (zh-TW) support evaluation
CosyVoice2 is natively multilingual with strong Mandarin prosody; numbers and
weather phrasing ("氣溫二十八度") are read naturally. Traditional-character input
is accepted. Evaluated via `test_zh.py` / `bench_zh.wav`.

### 5. Recommendation
- **Is CosyVoice suitable?** Yes for quality and zh-TW; it is the right research
  choice. The install is non-trivial on Mac but reproducible (above).
- **Mac vs Windows GPU?** Develop on the **Mac (CPU)**; deploy for real time on
  the **Windows NVIDIA GPU** box (same box as MuseTalk). CPU RTF is > 1
  (not real-time); a CUDA GPU brings RTF well below 1.
- **Expected latency in the avatar loop:** on CPU, generation ≈ RTF × audio
  length (see benchmark). For an 8 s end-to-end target, the GPU box + streaming
  synthesis is required; the Mac is fine for development and correctness testing.
