"""MOSS-TTS-Realtime STREAMING TTS server -- same wire contract as cosyvoice-local-tts
(POST /tts/stream {text, voice, sample_rate} -> raw 16-bit PCM mono stream), so the
pipeline reaches it through the existing CosyVoice client just by repointing the URL.

STREAMING: uses MOSS's MossTTSRealtimeStreamingSession + AudioStreamDecoder
(push_text -> decode -> end_text -> drain -> flush) so the FIRST audio chunk is emitted
while the rest still generates -- time-to-first-audio drops from ~full-sentence (the old
non-streaming server: TTFB == total gen time) to roughly the first-chunk time. This is
the real fix for the lip-start delay; the 1.7B param size only affects steady-state RTF,
not this first-chunk wait.

Runs in the `moss-tts` conda env. MUST be launched with LD_LIBRARY_PATH covering
torch/lib + the env lib + the nvidia/* pip libs or torchcodec can't dlopen (ffmpeg7 +
nvidia-npp fix). Example:

  conda activate moss-tts
  SP=$CONDA_PREFIX/lib/python3.12/site-packages
  export LD_LIBRARY_PATH=$SP/torch/lib:$CONDA_PREFIX/lib:$(ls -d $SP/nvidia/*/lib|tr '\n' ':')
  export CC=$(ls $CONDA_PREFIX/bin/*-gcc|head -1)   # triton/torch.compile needs a C compiler
  export CXX=$(ls $CONDA_PREFIX/bin/*-g++|head -1)   # (conda install -n moss-tts -c conda-forge gcc gxx)
  export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/moss_inductor  # persist compiled kernels across restarts
  export MOSS_REALTIME_DIR=$HOME/ttsdemo/MOSS-TTS/moss_tts_realtime
  export MOSS_REF=/mnt/e/Claude/visualllm/assets/moss_pro_ref.wav
  python -m uvicorn local_services.moss_server.app:app --host 0.0.0.0 --port 8003

The voice is the fixed reference clip pinned by MOSS_REF (MOSS-Realtime is clone-only).

LATENCY: streaming yields the first audio chunk at ~0.3-0.5s TTFB once warm (vs ~8.5s for
a non-streaming synthesize-then-send server). The catch is torch.compile recompiles on
each NEW token-length the first time it sees it (~40s that once); the startup warmup runs
a spread of lengths so dynamo marks the length dim dynamic and live turns stay fast. CC +
the persistent inductor cache make this survive restarts. The no-recompile production path
is vLLM-Omni (MOSS supports it natively) -- the report's next step.
"""
from __future__ import annotations

import os

# Run the codec/model EAGER (no torch.compile) by default. Compiled mode is ~1.6x faster
# at steady state BUT recompiles ~3-40s the first time it sees each new sentence-length
# bucket -- which in a live varied-length conversation fires repeatedly and reads as "delay
# between sentences". Eager has no recompiles: every sentence is a consistent ~0.4s TTFB
# (the perceived-smoothness win), at the cost of slower long-sentence steady-state. Set
# MOSS_COMPILE=1 to opt back into compiled mode. The both-fast-and-no-spikes path is
# vLLM-Omni (MOSS supports it natively) -- the report's next step.
if os.environ.get("MOSS_COMPILE", "0") != "1":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

CODEC_SR = 24000
CHUNK_DURATION = 0.24
DECODE_CHUNK_FRAMES = 3   # lower = lower first-chunk latency, more decode overhead
MOSS_REALTIME_DIR = os.environ.get(
    "MOSS_REALTIME_DIR", os.path.expanduser("~/ttsdemo/MOSS-TTS/moss_tts_realtime")
)
MOSS_REF = os.environ.get("MOSS_REF", "/mnt/e/Claude/visualllm/assets/moss_pro_ref.wav")
sys.path.insert(0, MOSS_REALTIME_DIR)

_E: dict = {}
_LOCK = threading.Lock()   # serialize turns (batch_size=1, single-client avatar anyway)


def _load():
    import torchaudio
    from transformers import AutoModel, AutoTokenizer
    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor
    from mossttsrealtime.streaming_mossttsrealtime import (
        AudioStreamDecoder,
        MossTTSRealtimeInference,
        MossTTSRealtimeStreamingSession,
    )

    import importlib.util
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    print(f"[moss] loading MOSS-TTS-Realtime streaming (attn={attn}) ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("OpenMOSS-Team/MOSS-TTS-Realtime")
    processor = MossTTSRealtimeProcessor(tokenizer)
    model = MossTTSRealtime.from_pretrained(
        "OpenMOSS-Team/MOSS-TTS-Realtime", attn_implementation=attn, torch_dtype=dtype
    ).to(device)
    model.eval()
    codec = AutoModel.from_pretrained(
        "OpenMOSS-Team/MOSS-Audio-Tokenizer", trust_remote_code=True
    ).eval().to(device)

    # Encode the fixed reference voice once.
    with torch.inference_mode():
        wav, sr = torchaudio.load(MOSS_REF)
        if sr != CODEC_SR:
            wav = torchaudio.functional.resample(wav, sr, CODEC_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        enc = codec.encode(wav.unsqueeze(0).to(device), chunk_duration=CHUNK_DURATION)
        prompt_tokens = enc["audio_codes"].cpu().numpy().squeeze(1)

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=3000)
    inferencer.reset_generation_state(keep_cache=False)
    session = MossTTSRealtimeStreamingSession(
        inferencer, processor, codec=codec, codec_sample_rate=CODEC_SR,
        codec_encode_kwargs={"chunk_duration": CHUNK_DURATION},
        prefill_text_len=processor.delay_tokens_len,
        temperature=0.8, top_p=0.6, top_k=30, do_sample=True,
        repetition_penalty=1.1, repetition_window=50,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    # Fixed-voice input_ids prefix (system prompt + assistant turn opener); reused each turn.
    system_prompt = processor.make_ensemble(prompt_tokens)
    pref_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    pref = np.full((len(pref_ids), system_prompt.shape[1]),
                   fill_value=processor.audio_channel_pad, dtype=np.int64)
    pref[:, 0] = pref_ids
    input_ids = np.concatenate([system_prompt, pref], axis=0)

    _E.update(dict(
        device=device, codec=codec, session=session, decoder_cls=AudioStreamDecoder,
        input_ids=input_ids, codebook_size=int(getattr(codec, "codebook_size", 1024)),
        audio_eos=int(getattr(session.inferencer, "audio_eos_token", 1026)),
    ))
    print(f"[moss] streaming ready (ref={MOSS_REF})", flush=True)


def _sanitize(tokens, codebook_size, eos):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
        return tokens
    eos_rows = (tokens[:, 0] == eos).nonzero(as_tuple=False)
    invalid = ((tokens < 0) | (tokens >= codebook_size)).any(dim=1)
    stop = int(eos_rows[0].item()) if eos_rows.numel() > 0 else None
    if invalid.any():
        iv = int(invalid.nonzero(as_tuple=False)[0].item())
        stop = iv if stop is None else min(stop, iv)
    return tokens[:stop] if stop is not None else tokens


def _decode_frames(frames, decoder) -> Iterator[np.ndarray]:
    cb, eos = _E["codebook_size"], _E["audio_eos"]
    for frame in frames:
        tok = frame
        if tok.dim() == 3:
            tok = tok[0]
        tok = _sanitize(tok, cb, eos)
        if tok.numel() == 0:
            continue
        decoder.push_tokens(tok.detach())
        for wav in decoder.audio_chunks():
            if wav.numel():
                yield wav.detach().cpu().numpy().reshape(-1)


def _synth_stream(text: str) -> Iterator[np.ndarray]:
    """Yield float32 mono 24 kHz audio chunks AS THEY GENERATE (streaming)."""
    session, codec = _E["session"], _E["codec"]
    session.inferencer.reset_generation_state(keep_cache=False)
    session.reset_turn(input_ids=_E["input_ids"], include_system_prompt=False, reset_cache=True)
    decoder = _E["decoder_cls"](
        codec, chunk_frames=DECODE_CHUNK_FRAMES, overlap_frames=0,
        decode_kwargs={"chunk_duration": -1}, device=_E["device"],
    )
    with torch.inference_mode(), codec.streaming(batch_size=1):
        yield from _decode_frames(session.push_text(text), decoder)
        yield from _decode_frames(session.end_text(), decoder)
        while True:
            frames = session.drain(max_steps=1)
            if not frames:
                break
            yield from _decode_frames(frames, decoder)
            if session.inferencer.is_finished:
                break
        final = decoder.flush()
        if final is not None and final.numel():
            yield final.detach().cpu().numpy().reshape(-1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    # torch.compile recompiles on each new token-length bucket; warm a spread of lengths
    # up front so a live conversation doesn't pay ~40s compile on the first turn of each
    # new sentence size. (The proper no-recompile path is vLLM-Omni -- the report's next
    # step; this warmup is the pragmatic fix for the plain-PyTorch server.)
    warm_texts = [
        "你好。",
        "你好，今天天气怎么样？",
        "你好，我是你的智能虚拟助手，很高兴认识你。",
        "你好，我是你的智能虚拟助手，今天天气真不错，有什么我可以帮你的吗？",
        "好的，没问题，我很乐意帮助你，请告诉我你想了解的内容，我会尽力为你详细解答清楚。",
    ]
    t0 = time.perf_counter()
    for i, wt in enumerate(warm_texts):
        try:
            for _ in _synth_stream(wt):
                pass
            print(f"[moss] warm {i+1}/{len(warm_texts)} (len {len(wt)})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[moss] warm {i+1} skipped: {e}", flush=True)
    print(f"[moss] warmup done in {time.perf_counter() - t0:.1f}s", flush=True)
    yield


app = FastAPI(title="MOSS-TTS-Realtime streaming TTS", version="2.0", lifespan=lifespan)


class TTSStreamRequest(BaseModel):
    text: str
    voice: str = Field("pro", description="informational; the engine uses MOSS_REF")
    sample_rate: int = 24000


@app.get("/health")
def health():
    return {"ok": "session" in _E}


@app.get("/")
def root():
    return {"service": "MOSS-TTS-Realtime (streaming)", "endpoints": ["/tts/stream (POST)", "/health"],
            "ref": MOSS_REF}


def _to_pcm16(chunk: np.ndarray, target_sr: int) -> bytes:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if target_sr != CODEC_SR:
        import librosa
        chunk = librosa.resample(chunk, orig_sr=CODEC_SR, target_sr=target_sr)
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()


@app.post("/tts/stream")
def tts_stream(req: TTSStreamRequest):
    if "session" not in _E:
        return JSONResponse({"error": "engine not ready"}, status_code=503)

    def gen():
        # Hold the lock for the whole turn so concurrent requests don't corrupt the
        # single streaming session (the avatar is single-client, so this is fine).
        with _LOCK:
            for chunk in _synth_stream(req.text):
                pcm = _to_pcm16(chunk, req.sample_rate)
                if pcm:
                    yield pcm

    return StreamingResponse(gen(), media_type="audio/L16")
