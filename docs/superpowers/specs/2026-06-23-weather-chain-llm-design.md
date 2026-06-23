# Design: Weather-chain LLM node + local memory harness

_2026-06-23 — baseline `cd88f20`. Status: approved (scope + approach), pending spec review._

## Goal

Two coupled requirements from the collaborator:

1. **Replace the general OpenRouter LLM** at the pipeline's LLM slot with a **dedicated Chinese
   weather assistant** that calls a LangServe weather-chain endpoint. Every user turn becomes a
   weather question routed to the chain; the chain's streamed answer flows through the unchanged
   TTS → MuseTalk avatar path. The avatar becomes a spoken Mandarin weather bot.
2. **Give the virtual human growing, persistent memory** ("context engineering + harness
   engineering … increase the memory continuously after a conversation") — a memory layer we
   build *around* the stateless chain, kept **fully local** (local `qwen2.5:3b`, no cloud).

The endpoint (NCU, Taiwan):

```
POST http://140.115.54.87:8000/chain/resWeatherChain/stream
Content-Type: application/json
{"input": {"query": "明天台北市有下雨嗎?", "model": "gemma3:27b"}}
```

It is a LangServe `add_routes` Runnable `/stream` route → Server-Sent Events (SSE). The chain
**requires the query in Chinese** and answers in Chinese.

## Why this is a clean swap

In pipecat 1.3.0 the LLM slot only owes the rest of the pipeline one contract (verified in
`pipecat/services/openai/base_llm.py::process_frame`): on each inbound `LLMContextFrame`, emit

```
LLMFullResponseStartFrame → LLMTextFrame(chunk)* → LLMFullResponseEndFrame
```

TTS sentence-aggregation, the avatar, and `aggregator.assistant()` all key off those frames and
do not care about the source. So swapping the brain touches **only** the LLM stage — nothing
downstream changes.

## Approach (chosen: A — custom `LLMService` subclass)

A new `WeatherChainLLMService(LLMService)` that speaks pipecat on the outside and the weather
chain on the inside. Rejected alternatives: **B** (bare `FrameProcessor` on raw transcripts —
bypasses the aggregator's turn/echo-guard handling and forces hand-rolled start/end markers, a
fragile foot-gun) and **C** (a local OpenAI-compatible shim process — a 4th server to babysit and
*two* format translations instead of one; the proxy pattern only earns its keep across many tools).

## Components

### 1. `local_services/weather_chain_llm.py` — `WeatherChainLLMService(LLMService)`

New file (services live in `local_services/`, like `cosyvoice_tts.py`). Subclasses pipecat's
`pipecat.services.llm_service.LLMService` so it inherits the conveyor-belt etiquette
(metrics, start/stop, error plumbing).

- **`process_frame`** — mirror `BaseOpenAILLMService.process_frame`: on `LLMContextFrame`, push
  `LLMFullResponseStartFrame()`, `start_processing_metrics()`, call `_process_context(context)`,
  and in `finally` push `LLMFullResponseEndFrame()` + `stop_processing_metrics()`. Pass every other
  frame through unchanged.
- **Query extraction** — from `context.get_messages()`, take the **last `role == "user"`
  message**'s text as `query`. The chain is stateless and weather-only — no history is sent (the
  system prompt and prior turns are irrelevant to it).
- **The call** — `httpx.AsyncClient` streaming POST to `{WEATHER_CHAIN_URL}/stream` with
  `{"input": {"query": query, "model": WEATHER_CHAIN_MODEL}}`. Reuse one client instance across
  turns (open it lazily; close on `stop`/`cancel`).
- **SSE parse** — read line-by-line. LangServe `/stream` emits `event: <name>` / `data: <json>`
  pairs. For `event: data`, JSON-decode the `data:` payload and extract text **tolerantly**: the
  chunk may be a bare JSON string (`"明天"`), or an object with a `content`/`output` field
  (depends on the chain's output parser). Push each non-empty piece as `LLMTextFrame`. Stop on
  `event: end` (or stream close).
- **Errors** — on connection failure / HTTP error / timeout: `push_error(...)` and emit one short
  spoken Chinese fallback (e.g. "抱歉，天氣服務暫時連線不上。") as an `LLMTextFrame` so the avatar
  isn't silently dead. Never raise out of `process_frame` (the base wraps it, but keep the spoken
  fallback explicit).

### 2. `pipeline/config.py` — provider switch + chain knobs

Keep OpenRouter as a deliberate **fallback switch** (the repo's convention, not multi-provider
branching), exactly like `TTS_PROVIDER`:

- `llm_provider: str = _get("LLM_PROVIDER", "openrouter")` → `openrouter` | `weather_chain`.
- `weather_chain_url: str = _get("WEATHER_CHAIN_URL", "http://140.115.54.87:8000/chain/resWeatherChain")`
  — base; the service appends `/stream`.
- `weather_chain_model: str = _get("WEATHER_CHAIN_MODEL", "gemma3:27b")`.

### 3. `pipeline/stages/llm.py` — factory branches on provider

`build_llm(cfg)` returns `WeatherChainLLMService(...)` when `cfg.llm_provider == "weather_chain"`,
else the existing `OpenAILLMService`. Thin, single-provider per branch — matches the other
factories. Import the weather service lazily inside the branch (keep preflight import-clean).

### 4. `pipeline/main.py` — warmup guard + greeting

- `_warmup_llm()` currently calls `llm._client.chat.completions.create(...)` — OpenAI-only; it
  would crash on the weather service. Guard it: only run the chat-completions warmup when the
  service exposes that client (e.g. `hasattr(llm, "_client")` / provider check); otherwise no-op
  (the chain has no cheap warmup ping).
- The startup log line and the greeting: when `weather_chain`, log the weather provider and use a
  Chinese weather-themed greeting (the existing `is_mandarin` greeting already covers Chinese;
  optionally tailor it to weather).

### 5. `.env` for the demo + `scripts/` probe

- `.env`: `LLM_PROVIDER=weather_chain`, `LANGUAGE=zh` (Deepgram zh-TW in, CosyVoice female-Mandarin
  out — already the natural fit). Document the new knobs in `.env` / WORKFLOW.md §8.
- `scripts/probe_weather_chain.py` — a tiny standalone POST that prints the raw SSE bytes, so the
  exact chunk shape can be confirmed the instant the server is reachable and the parser tweaked in
  one line if LangServe's framing differs from the assumed `event: data` / `data: <json-string>`.

## Memory harness (requirement 3) — fully local

The chain accepts **only** `{"query","model"}`; we don't own it and can't feed it history or a
profile. So the virtual human's memory lives entirely in **our harness, wrapped around** the
stateless chain. The engine for the memory ops is **local `qwen2.5:3b` via Ollama's
OpenAI-compatible endpoint** (`http://localhost:11434/v1`) — no cloud, no per-token cost, matching
the local-first stack (CosyVoice/MuseTalk). Validated 2026-06-23 (qwen resolved both continuity
and profile-fill correctly in Traditional Chinese — see §"Validation").

### 6. `local_services/avatar_memory.py` — `MemoryStore` (harness engineering)

Persists to `AVATAR_MEMORY_DIR` (default `state/avatar_memory/`, gitignored — per-user runtime):
- `profile.json` — durable facts: `name`, `default_city`, `preferences`, free-form `notes`. Small,
  always loaded. **Single shared profile** (one viewer at a time; per-user keying is out of scope).
- `summary.txt` — a rolling Traditional-Chinese summary of past conversations (long-term memory).
- `session.jsonl` — the current conversation's turns (`{user, bot, ts}`), appended live.

API: `recall() -> str` (compact zh context block = profile + summary), `record_turn(user, bot)`,
`async distill(client)` (end-of-conversation profile+summary update).

### 7. Turn-time context engineering — gated query rewrite

Inside `WeatherChainLLMService`, before the chain call: take the raw utterance + `memory.recall()`
and, **gated**, rewrite it into a self-contained zh weather query via the local qwen client.
- **Gated** (`MEMORY_LLM_GATED=1`, default): only call qwen when the utterance looks
  context-dependent (pronoun/ellipsis like "那…呢", or names no city). Otherwise pass the utterance
  straight through. Keeps the fast path fast and **minimizes GPU contention** with the avatar
  render (the project's #1 smoothness enemy). `MEMORY_LLM_GATED=0` = always rewrite.
- Prompt shape (validated): **single user turn, few-shot, ending in `改寫：`** as a completion
  primer. A system-prompt-heavy shape did not steer the 3B reliably.
- After the turn: `memory.record_turn(raw, answer)`.

### 8. After-conversation distillation (the *continuous growth*)

`main.py::on_client_disconnected` (already fires at conversation end): `await memory.distill(client)`
— one local-qwen call reads the session turns + old profile/summary and returns (a) a merged
profile and (b) a refreshed rolling summary; persist both, clear `session.jsonl`. **This is the
step that grows the human's memory after every chat.** Not latency-critical (post-conversation) →
GPU contention is irrelevant here. Next connect, the greeting is personalized from `profile.json`.

### 9. Config + degradation (hardening)

New config (all `.env`): `AVATAR_MEMORY` (1=on, default), `MEMORY_LLM_URL`
(`http://localhost:11434/v1`), `MEMORY_LLM_MODEL` (**`qwen2.5:3b-cpu`**, default), `MEMORY_LLM_GATED`
(1), `AVATAR_MEMORY_DIR` (`state/avatar_memory`). **Degrade cleanly**: if Ollama is down or a
rewrite/distill call fails/times out, pass the raw utterance through and skip distillation — the
weather bot still works, just without memory that turn. Memory never blocks or breaks a turn.

**CPU-pinned by default (decided + measured 2026-06-23).** The memory model runs on **CPU**, not
the GPU, so MuseTalk + CosyVoice keep the whole 16 GB card. One-time setup (done on this box):
```
printf 'FROM qwen2.5:3b\nPARAMETER num_gpu 0\n' > qwen-cpu.Modelfile
ollama create qwen2.5:3b-cpu -f qwen-cpu.Modelfile
```
Measured: gated rewrite **0.77 s** on CPU, `ollama ps` = `100% CPU`, output identical to GPU. To
move it back to GPU (if the CPU ever bottlenecks), just set `MEMORY_LLM_MODEL=qwen2.5:3b`. Distill
is slower on CPU (~10–20 s) but runs post-disconnect, off the hot path.

## Data flow (one turn, with memory)

```
mic → STT(zh) → aggregator.user() → LLMContextFrame
   → WeatherChainLLMService:
        raw utterance + memory.recall()
        → [gated] local qwen2.5:3b rewrite → self-contained zh query
        → POST chain/stream → SSE tokens → LLMTextFrame*
        → memory.record_turn(raw, answer)
   → TTS(CosyVoice) → MuseTalk avatar → browser
   aggregator.assistant() records the bot turn
… on disconnect: memory.distill(qwen) → profile.json + summary.txt grow
```

## Risks / flags (not hidden)

- **Server unreachable now.** Port 8000 refuses connection (host pings, 78 ms). Built to the
  standard LangServe `/stream` contract; the probe + a possible 1-line SSE-parse tweak close the
  gap once it's up. **The SSE chunk shape is an assumption until verified.**
- **TTFO < 8 s may not hold.** `gemma3:27b` on Ollama in Taiwan + a likely weather-retrieval step
  may be slower than OpenRouter; a gated rewrite adds one short local-qwen hop on context-dependent
  turns. Streaming first-token mitigates; measure with `scripts/measure`.
- **GPU contention — mitigated by the CPU-pin default.** The 16 GB card is already ~13.4/16 used
  (MuseTalk + CosyVoice-vLLM). Rather than risk a rewrite competing with the render, the memory
  model is CPU-pinned (`qwen2.5:3b-cpu`, `num_gpu 0`) — measured 0.77 s/rewrite, zero GPU use. GPU
  is available as an opt-in (`MEMORY_LLM_MODEL=qwen2.5:3b`) only if the CPU ever bottlenecks.
- **Windows UTF-8 trap (verified).** Inline `curl -d` with Chinese, and `print()` to the cp1252
  console, both corrupt the bytes (caused a false "model can't read Chinese" scare). All Chinese
  must travel as real UTF-8 (httpx/JSON `.encode("utf-8")`); never inline-curl Chinese on Windows.

## Out of scope

Intent routing (weather vs general), tool-calling, per-user memory keying, vector/embedding
retrieval (the profile+summary is small enough to always load), and any change to STT/TTS/avatar/
sync. Reverting to plain general chat is a one-line `.env` flip (`LLM_PROVIDER=openrouter`); memory
off is `AVATAR_MEMORY=0`.

## Validation (done 2026-06-23, before build)

Local `qwen2.5:3b` via Ollama `/v1` (clean UTF-8 over httpx):
- continuity: "那台中呢？" + (lives Taipei, prior "明天台北市會下雨嗎？") → **"明天台中會下雨嗎？"** ✓
- profile-fill: "明天會很熱嗎？" + (lives Kaohsiung) → **"明天高雄市會很熱嗎？"** ✓
The earlier garbled outputs were the Windows UTF-8 trap, not the model.

## Testing / verification

- `python -m scripts.preflight` — imports resolve with no keys/network.
- `python scripts/probe_weather_chain.py` — confirm live SSE shape (when reachable).
- A small memory unit test: seed `profile.json`, feed an elliptical utterance, assert the rewrite
  fills it; run `distill` over a fake session, assert profile/summary update + `session.jsonl` clear.
- Live: `LANGUAGE=zh LLM_PROVIDER=weather_chain AVATAR_MEMORY=1`, open `/client/`, ask a Chinese
  weather question + a follow-up ("那台中呢?"), confirm the avatar resolves it; disconnect, reopen,
  confirm the greeting/profile reflects the prior chat. Check `[TTFO]` + `scripts/measure`.
