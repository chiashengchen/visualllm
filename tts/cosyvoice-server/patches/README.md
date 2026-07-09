# CosyVoice zh fixes (apply to the upstream clone)

The upstream `CosyVoice/` clone (Alibaba's repo + the CosyVoice2-0.5B weights) is **not**
committed here — it's `.gitignore`d (~5 GB). These two small patches restore correct Chinese
behaviour on the vLLM decode path and must be applied to that clone after you download it.

## Why (root cause)

Running CosyVoice2's autoregressive speech-token LLM on **vLLM** (the latency fix) bypassed the
model's own **Repetition-Aware Sampling (RAS)** — vLLM does its own sampling. Without RAS the LLM
intermittently **loops on the silence token**: the same short Chinese sentence comes out ~4 s clean
one run and **~12 s with ~5 s of dead silence** the next (~40 % of zh runs; English is unaffected —
it uses the `cross_lingual` path). That silence is heard as "halting/broken" zh, and the talking-head
avatar keeps animating through it.

vLLM's built-in `repetition_penalty`/`frequency_penalty` **cannot** be used to fix this: they build a
prompt-token bincount, but CosyVoice feeds `prompt_embeds` (no prompt token ids) → a CUDA
`ScatterGatherKernel index out of bounds` device-side assert that kills the engine.

## The fix (two files)

1. **`ras_logits_processor.py`** → copy to `CosyVoice/cosyvoice/vllm/ras_logits_processor.py`.
   A vLLM V1 per-request logits processor that bans any token seen in the last `COSYVOICE_RAS_WIN`
   (=10) **output** tokens — RAS's anti-loop rule, using output tokens only (embeds-safe).
2. **`cosyvoice_vllm_ras.patch`** → registers the processor via
   `EngineArgs(logits_processors=[RasLogitsProcessor])` in `cosyvoice/cli/model.py::load_vllm`, and
   adds `top_p=0.8` to `cosyvoice/llm/llm.py::inference_wrapper` (matches RAS's nucleus).

```bash
# from tts/cosyvoice-server/
cp patches/ras_logits_processor.py CosyVoice/cosyvoice/vllm/ras_logits_processor.py
git -C CosyVoice apply ../patches/cosyvoice_vllm_ras.patch    # or: patch -p1 -d CosyVoice < patches/cosyvoice_vllm_ras.patch
# the default "pro" reference voice (tts_engine.py expects it at CosyVoice/asset/pro_ref.wav):
cp ../../assets/moss_pro_ref.wav CosyVoice/asset/pro_ref.wav
```

Verify: driving varied zh sentences at `/tts/stream` should give **0 degenerate** (no >7 s / all-silence
outputs). Knob: `COSYVOICE_RAS_WIN` (default 10; 0 disables).

## Voice (in `tts_engine.py`, no patch needed)

The default zero-shot reference is the fluid **"pro" AI-assistant voice** (`assets/moss_pro_ref.wav` in
the repo root; transcript `你好，我是你的AI虚拟助手，很高兴见到你。今天天气不错，有什么我可以帮你的`).
`zero_shot` clones the reference's *rhythm*, and this clip is naturally smooth, so zh ends up ≈ English
pacing (~1 pause/sentence) with no trimming. Override with `COSYVOICE_PROMPT_WAV`/`COSYVOICE_PROMPT_TEXT`.
An optional zh pause-trimmer (`COSYVOICE_SILENCE_CAP_S`, `_squeeze_silence`) is **off by default**.

Full write-up: `docs/PROBLEMS-AND-FIXES.md` P18 in the repo root.
