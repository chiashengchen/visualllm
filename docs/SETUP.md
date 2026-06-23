# Setup — buying & wiring the APIs (dedicated provider per stage)

> **⚠️ Partly historical (the original cloud-API setup guide).** The current default stack is
> mostly **local**: TTS = **CosyVoice2 on vLLM in WSL** (not ElevenLabs), avatar = **MuseTalk**
> local server (not Simli). The only paid APIs still required are **Deepgram** (STT) and
> **OpenRouter** (LLM); ElevenLabs is now an optional fallback. See **`STATUS.md`**
> for the live stack and **`WORKFLOW.md` §8** for the `.env` reference. The provider-shopping
> details below are kept for the cloud-fallback path.

We use a **dedicated streaming provider for each stage** (not an aggregator like
fal.ai), because streaming providers give the lowest latency — which is what the
<8 s time-to-first-output goal needs. You buy four services; each owns one box in
the pipeline.

```
speech → [Deepgram] → [OpenRouter/OpenAI] → [ElevenLabs] → [Simli] → video+audio
            STT            LLM                  TTS           avatar
```

---

## What to buy (Phase 1, English)

| # | Service | Stage | Sign up | Free tier | Rough cost after |
|---|---------|-------|---------|-----------|------------------|
| 1 | **Deepgram** | STT | console.deepgram.com | $200 credit | ~$0.0043 / min (nova) |
| 2 | **OpenRouter** *or* **OpenAI** | LLM | openrouter.ai / platform.openai.com | pay-as-you-go | gpt-4o-mini ≈ $0.15–0.60 / 1M tok |
| 3 | **ElevenLabs** | TTS | elevenlabs.io | ~10k chars/mo | from ~$5/mo |
| 4 | **Simli** | Avatar | simli.com | free dev tier | usage-based |

All four have a free/credit tier, so the **full pipeline can be proven for $0**.

---

## The values you copy (key vs. ID)

A **key** authenticates *you*. Some services also need an **ID** that picks
*which* voice/face/model to use.

| Service | Key | Extra ID | Where the ID comes from |
|---------|-----|----------|--------------------------|
| Deepgram | `DEEPGRAM_API_KEY` | — | — |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` | model string, e.g. `openai/gpt-4o-mini` |
| OpenAI (alt) | `OPENAI_API_KEY` | — | — |
| ElevenLabs | `ELEVENLABS_API_KEY` | `ELEVENLABS_VOICE_ID` | pick a voice in Voice Library → "Copy Voice ID" |
| Simli | `SIMLI_API_KEY` | `SIMLI_FACE_ID` | pick/create a face in the dashboard |

---

## Step by step

1. **Create the 4 accounts** above and copy each key (and the ElevenLabs voice ID
   + Simli face ID — just grab default/prebuilt ones to start).
2. `copy .env.example .env`
3. Fill in `.env`. Pick **one** LLM route:

   **Option A — OpenRouter (one key, swap models freely):**
   ```
   STT_PROVIDER=deepgram
   LLM_PROVIDER=openrouter
   TTS_PROVIDER=elevenlabs
   AVATAR_PROVIDER=simli
   LANGUAGE=en

   DEEPGRAM_API_KEY=...
   OPENROUTER_API_KEY=...
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ELEVENLABS_API_KEY=...
   ELEVENLABS_VOICE_ID=...
   SIMLI_API_KEY=...
   SIMLI_FACE_ID=...
   ```

   **Option B — OpenAI direct:** set `LLM_PROVIDER=openai` and fill
   `OPENAI_API_KEY` instead of the OpenRouter lines.

4. **Verify before running** (catches any missing key / import drift):
   ```
   python -m scripts.preflight
   ```
   Selected stages should flip from `KEYS` to `PASS`.
5. **Run:**
   ```
   python -m pipeline.main
   ```
   Open the printed `http://localhost:7860`, allow the mic, and talk. Watch the
   `[TTFO]` lines — that's your time-to-first-output per turn.

---

## Phase 2 — Mandarin (no new accounts needed)

Same four services handle Chinese; only `.env` values change:

```
LANGUAGE=zh
OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct   # or deepseek/deepseek-chat — strong zh
ELEVENLABS_VOICE_ID=<a voice that sounds good in Mandarin>
```

- **STT:** Deepgram supports `zh-TW` (already handled in code when `LANGUAGE=zh`).
- **LLM:** via OpenRouter you can A/B Qwen / DeepSeek / Gemini for Mandarin with a
  one-line model change — no extra key.
- **TTS:** ElevenLabs `eleven_flash_v2_5` is multilingual; just choose a
  zh-friendly voice ID. (Later, swap to local CosyVoice2 in Phase 3 for quality.)
- **Avatar:** Simli lip-syncs zh audio unchanged.

---

## Which keys are NOT needed

- **No aggregator** (fal.ai / Replicate) — we buy per stage for latency.
- **Phase 2/3 local models** (FunASR, CosyVoice2, MuseTalk, local Qwen) use
  **downloaded checkpoints on the 5060 Ti, not API keys** — zero new keys.

So the complete shopping list is just the **four services above**.
