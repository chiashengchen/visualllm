# Weather-chain LLM + local memory harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the OpenRouter LLM with a dedicated Chinese weather bot backed by a remote LangServe weather chain, wrapped in a fully-local (CPU qwen2.5:3b) memory harness that grows the virtual human's memory after every conversation.

**Architecture:** A new `WeatherChainLLMService(LLMService)` drops into the existing pipeline slot — it consumes `LLMContextFrame` and emits `LLMFullResponseStart/Text/End` frames, so TTS/avatar/aggregator are untouched. A `MemoryStore` (separate module) persists a profile + rolling summary + session log to disk, rewrites the user's utterance into a self-contained query via local qwen before the chain call (context engineering), and distills the conversation into durable memory on disconnect. A `LLM_PROVIDER` switch keeps OpenRouter as a one-flip fallback.

**Tech Stack:** Python 3.11 (system env), Pipecat 1.3.0, httpx 0.28, Ollama (`qwen2.5:3b-cpu`, OpenAI-compatible at `:11434/v1`), remote LangServe chain.

**Spec:** `docs/superpowers/specs/2026-06-23-weather-chain-llm-design.md`.

## Global Constraints

- **No pytest in this repo.** Tests are standalone scripts in `archive/`, run as `python -m archive._<name>_test`, using plain `assert` + a `main()` and `if __name__ == "__main__": main()`. Match `archive/_frame_deficit_repro_test.py`.
- **Run `python -m scripts.preflight` after touching** `pipeline/stages/*.py`, `pipeline/main.py`, or `pipeline/metrics.py` — Pipecat import paths drift between releases.
- **UTF-8 everywhere; never print/log raw Chinese to the console.** The Windows console is cp1252 and crashes on Chinese (verified during design). All Chinese travels as JSON bytes over httpx (`json=` handles encoding). Log only ASCII (lengths, model names, error types) — never the Chinese text itself.
- **Memory must never break or block a turn.** Every memory call (`build_query`, `record_turn`, `distill`) is wrapped so any failure falls back to pass-through; the weather bot keeps working without memory.
- **Stage factories stay thin + single-provider; config is `.env`-driven only.** Comments state the *why* (latency / Pipecat quirk / hardware), matching the house voice.
- **Reverting is one flag:** `LLM_PROVIDER=openrouter` (general chat), `AVATAR_MEMORY=0` (memory off).

## File structure

- **Create** `local_services/weather_chain_llm.py` — `WeatherChainLLMService` + the pure `extract_sse_text()` helper.
- **Create** `local_services/avatar_memory.py` — `MemoryStore` (persistence + recall + record + distill) and the pure `needs_rewrite()` gating helper.
- **Modify** `pipeline/config.py` — add the provider switch + chain + memory knobs.
- **Modify** `pipeline/stages/llm.py` — branch `build_llm(cfg, memory)` on provider.
- **Modify** `pipeline/main.py` — create the `MemoryStore`, pass it into `build_llm`, guard `_warmup_llm`, personalize the greeting, distill on disconnect.
- **Create** `scripts/probe_weather_chain.py` — standalone SSE-shape probe for when the chain is reachable.
- **Create** tests in `archive/`: `_sse_parse_test.py`, `_memory_store_test.py`, `_memory_gating_test.py`, `_llm_factory_test.py`, `_memory_rewrite_test.py` (live, needs Ollama).
- **Modify** `.env.example` (+ `WORKFLOW.md` §8 if present) — document the new knobs.
- **Setup (already done on this box, document it):** the `qwen2.5:3b-cpu` Ollama model.

---

### Task 1: Config knobs (provider switch + chain + memory)

**Files:**
- Modify: `pipeline/config.py` (inside the `Config` dataclass, after the OpenRouter block ~line 61)
- Test: `archive/_config_knobs_test.py`

**Interfaces:**
- Produces: `config.llm_provider: str`, `config.weather_chain_url: str`, `config.weather_chain_model: str`, `config.avatar_memory: bool`, `config.avatar_memory_dir: str`, `config.memory_llm_url: str`, `config.memory_llm_model: str`, `config.memory_llm_gated: bool`.

- [ ] **Step 1: Write the failing test**

```python
# archive/_config_knobs_test.py
"""The weather-bot + memory config knobs exist with the documented defaults.
Run: python -m archive._config_knobs_test"""
import dataclasses
from pipeline.config import Config


def main() -> None:
    c = Config()
    assert c.llm_provider in ("openrouter", "weather_chain"), c.llm_provider
    assert c.weather_chain_url.startswith("http"), c.weather_chain_url
    assert c.weather_chain_model, "weather_chain_model empty"
    assert isinstance(c.avatar_memory, bool)
    assert c.memory_llm_url.endswith("/v1"), c.memory_llm_url
    assert "qwen" in c.memory_llm_model, c.memory_llm_model
    assert isinstance(c.memory_llm_gated, bool)
    # the switch is usable with dataclasses.replace (frozen dataclass)
    w = dataclasses.replace(c, llm_provider="weather_chain")
    assert w.llm_provider == "weather_chain"
    print("PASS _config_knobs_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._config_knobs_test`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'llm_provider'`

- [ ] **Step 3: Add the config fields**

In `pipeline/config.py`, immediately after the `openrouter_model` line (~line 61), insert:

```python
    # --- LLM provider switch (deliberate fallback switch, like TTS_PROVIDER) ---
    # weather_chain = a dedicated Chinese weather bot backed by the NCU LangServe
    # endpoint; openrouter = the general-chat fallback. One flip reverts.
    llm_provider: str = (_get("LLM_PROVIDER", "openrouter") or "openrouter").lower()
    weather_chain_url: str = _get(
        "WEATHER_CHAIN_URL", "http://140.115.54.87:8000/chain/resWeatherChain"
    )  # base; the service appends /stream
    weather_chain_model: str = _get("WEATHER_CHAIN_MODEL", "gemma3:27b") or "gemma3:27b"

    # --- Avatar memory harness (fully local: qwen2.5:3b on CPU via Ollama) ---
    # The chain is stateless, so the virtual human's growing memory lives here.
    # CPU-pinned (qwen2.5:3b-cpu) so MuseTalk + CosyVoice keep the whole GPU.
    avatar_memory: bool = (_get("AVATAR_MEMORY", "1") or "1").lower() in ("1", "true", "yes", "on")
    avatar_memory_dir: str = _get("AVATAR_MEMORY_DIR", "state/avatar_memory") or "state/avatar_memory"
    memory_llm_url: str = _get("MEMORY_LLM_URL", "http://localhost:11434/v1") or "http://localhost:11434/v1"
    memory_llm_model: str = _get("MEMORY_LLM_MODEL", "qwen2.5:3b-cpu") or "qwen2.5:3b-cpu"
    # Gated = only rewrite when the utterance looks context-dependent (keeps the
    # fast path fast; CPU rewrite ~0.77s when it does fire). 0 = always rewrite.
    memory_llm_gated: bool = (_get("MEMORY_LLM_GATED", "1") or "1").lower() in ("1", "true", "yes", "on")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m archive._config_knobs_test`
Expected: `PASS _config_knobs_test`

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py archive/_config_knobs_test.py
git commit -m "feat(config): add LLM_PROVIDER switch + weather-chain + memory knobs"
```

---

### Task 2: SSE text extractor (pure, tolerant)

**Files:**
- Create: `local_services/weather_chain_llm.py` (just the helper for now)
- Test: `archive/_sse_parse_test.py`

**Interfaces:**
- Produces: `extract_sse_text(data: str) -> Optional[str]` — given the text *after* `data:` on one SSE line, returns the text piece, or `None` for control/empty/non-text payloads.

- [ ] **Step 1: Write the failing test**

```python
# archive/_sse_parse_test.py
"""extract_sse_text tolerates the LangServe /stream chunk shapes we might see.
Run: python -m archive._sse_parse_test"""
from local_services.weather_chain_llm import extract_sse_text


def main() -> None:
    # bare JSON string chunk (most likely for a StrOutputParser chain)
    assert extract_sse_text('"明天"') == "明天"
    # object with a content field
    assert extract_sse_text('{"content": "台北"}') == "台北"
    # object with an output field
    assert extract_sse_text('{"output": "下雨"}') == "下雨"
    # raw non-JSON text (some chains stream plain text)
    assert extract_sse_text("hello") == "hello"
    # control / empty payloads -> None
    assert extract_sse_text("[DONE]") is None
    assert extract_sse_text("   ") is None
    assert extract_sse_text('{"event": "metadata"}') is None
    print("PASS _sse_parse_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._sse_parse_test`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_services.weather_chain_llm'`

- [ ] **Step 3: Create the file with the helper**

```python
# local_services/weather_chain_llm.py
"""LLM stage variant: a dedicated Chinese weather bot backed by a remote LangServe
weather-chain endpoint. It drops into the same pipeline slot as the OpenRouter LLM --
consumes LLMContextFrame, emits LLMFullResponseStart/Text/End -- so TTS, the avatar,
and the assistant aggregator are unchanged.

The chain accepts ONLY {"query","model"} (no history), so the virtual human's memory
lives in the optional MemoryStore wrapped around this service (see avatar_memory.py).
"""
from __future__ import annotations

import json
from typing import Optional


def extract_sse_text(data: str) -> Optional[str]:
    """Pull the text out of one LangServe SSE `data:` payload, tolerantly.

    LangServe /stream emits `event: data` + `data: <json>`. The json may be a bare
    string ("明天") or an object ({"content": ...} / {"output": ...}). Returns the text
    piece, or None for control payloads ([DONE], empty, metadata, unparseable-non-text).
    """
    s = data.strip()
    if not s or s == "[DONE]":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return s or None  # some chains stream raw text after `data: `
    if isinstance(obj, str):
        return obj or None
    if isinstance(obj, dict):
        for key in ("content", "output", "text", "answer"):
            v = obj.get(key)
            if isinstance(v, str) and v:
                return v
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m archive._sse_parse_test`
Expected: `PASS _sse_parse_test`

- [ ] **Step 5: Commit**

```bash
git add local_services/weather_chain_llm.py archive/_sse_parse_test.py
git commit -m "feat(weather): tolerant LangServe SSE text extractor"
```

---

### Task 3: MemoryStore persistence + recall

**Files:**
- Create: `local_services/avatar_memory.py` (storage half; LLM ops added in Task 5)
- Test: `archive/_memory_store_test.py`

**Interfaces:**
- Produces:
  - `class MemoryStore` with `__init__(self, *, base_dir, llm_url=None, llm_model=None, gated=True, enabled=True)`
  - `profile: dict` (keys: `name`, `default_city`, `preferences: list`, `notes: str`)
  - `summary: str`
  - `session: list[dict]` (each `{"user", "bot", "ts"}`)
  - `recall() -> str` — compact zh context block (profile + summary)
  - `record_turn(self, user: str, bot: str) -> None` — append to `session` + `session.jsonl`
  - `reset_session(self) -> None` — clear `session` + truncate `session.jsonl`
  - `greeting_hint(self) -> Optional[str]` — a zh greeting suffix from the profile, or None
  - `_save_profile()` / `_save_summary()` — write `profile.json` / `summary.txt`

- [ ] **Step 1: Write the failing test**

```python
# archive/_memory_store_test.py
"""MemoryStore persists profile/summary/session and recalls them.
Run: python -m archive._memory_store_test"""
import json
import tempfile
from pathlib import Path

from local_services.avatar_memory import MemoryStore


def main() -> None:
    d = tempfile.mkdtemp()
    m = MemoryStore(base_dir=d, enabled=True)

    # fresh store: empty recall, no greeting hint
    assert m.recall() == "" or isinstance(m.recall(), str)
    assert m.greeting_hint() is None

    # seed a profile + summary, persist, reload
    m.profile["default_city"] = "台北市"  # Taipei
    m.profile["name"] = "Ann"
    m.summary = "使用者常問台北天氣"
    m._save_profile()
    m._save_summary()

    m2 = MemoryStore(base_dir=d, enabled=True)
    assert m2.profile.get("default_city") == "台北市"
    assert "台北市" in m2.recall()
    assert m2.greeting_hint() is not None  # has a default_city -> personalized greeting

    # record_turn appends to session + jsonl
    m2.record_turn("今天天氣", "晴天")
    assert len(m2.session) == 1
    lines = Path(d, "session.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["user"] == "今天天氣"

    # reset clears both
    m2.reset_session()
    assert m2.session == []
    assert Path(d, "session.jsonl").read_text(encoding="utf-8") == ""

    # disabled store is inert
    off = MemoryStore(base_dir=tempfile.mkdtemp(), enabled=False)
    off.record_turn("a", "b")
    assert off.session == []
    print("PASS _memory_store_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._memory_store_test`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_services.avatar_memory'`

- [ ] **Step 3: Create the storage half of `avatar_memory.py`**

```python
# local_services/avatar_memory.py
"""Avatar memory harness (fully local). Persists the virtual human's growing memory --
a durable profile, a rolling Chinese summary, and the live session log -- and (Task 5)
rewrites utterances into self-contained queries + distills conversations via local qwen.

Storage layout under base_dir (default state/avatar_memory/, gitignored):
  profile.json   durable facts {name, default_city, preferences[], notes}
  summary.txt    rolling zh summary of past conversations
  session.jsonl  current conversation turns ({user, bot, ts})

Hardening: memory NEVER breaks a turn -- callers wrap these in try/except, and a
disabled store (enabled=False) is fully inert. ASCII-only logging (the console is cp1252).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger

_DEFAULT_PROFILE = {"name": "", "default_city": "", "preferences": [], "notes": ""}


class MemoryStore:
    def __init__(
        self,
        *,
        base_dir: str,
        llm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        gated: bool = True,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.base = Path(base_dir)
        self._llm_url = llm_url
        self._llm_model = llm_model
        self._gated = gated
        self.profile = dict(_DEFAULT_PROFILE)
        self.summary = ""
        self.session: list[dict] = []
        if self.enabled:
            self.base.mkdir(parents=True, exist_ok=True)
            self._load()

    # ---- paths ----
    @property
    def _profile_path(self) -> Path:
        return self.base / "profile.json"

    @property
    def _summary_path(self) -> Path:
        return self.base / "summary.txt"

    @property
    def _session_path(self) -> Path:
        return self.base / "session.jsonl"

    # ---- load / save ----
    def _load(self) -> None:
        try:
            if self._profile_path.exists():
                loaded = json.loads(self._profile_path.read_text(encoding="utf-8"))
                self.profile = {**_DEFAULT_PROFILE, **loaded}
            if self._summary_path.exists():
                self.summary = self._summary_path.read_text(encoding="utf-8").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"memory load failed ({type(e).__name__}); starting empty")

    def _save_profile(self) -> None:
        self._profile_path.write_text(
            json.dumps(self.profile, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_summary(self) -> None:
        self._summary_path.write_text(self.summary, encoding="utf-8")

    # ---- recall (context engineering input) ----
    def recall(self) -> str:
        """Compact zh context block fed to the rewrite/distill prompts."""
        bits = []
        city = self.profile.get("default_city")
        name = self.profile.get("name")
        prefs = self.profile.get("preferences") or []
        if name:
            bits.append(f"使用者名稱：{name}")          # user name
        if city:
            bits.append(f"使用者住在：{city}")          # lives in
        if prefs:
            bits.append("偏好：" + "、".join(map(str, prefs)))  # preferences
        if self.summary:
            bits.append(f"過往摘要：{self.summary}")        # past summary
        return "\n".join(bits)

    def greeting_hint(self) -> Optional[str]:
        """A zh greeting tail personalized from the profile, or None if nothing known."""
        city = self.profile.get("default_city")
        if city:
            return f"還是想看{city}的天氣嗎？"  # "Still want <city>'s weather?"
        return None

    # ---- session log ----
    def record_turn(self, user: str, bot: str) -> None:
        if not self.enabled:
            return
        turn = {"user": user, "bot": bot, "ts": time.time()}
        self.session.append(turn)
        with self._session_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def reset_session(self) -> None:
        self.session = []
        if self.enabled:
            self._session_path.write_text("", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m archive._memory_store_test`
Expected: `PASS _memory_store_test`

- [ ] **Step 5: Commit**

```bash
git add local_services/avatar_memory.py archive/_memory_store_test.py
git commit -m "feat(memory): MemoryStore persistence + recall + session log"
```

---

### Task 4: Rewrite-gating heuristic (pure)

**Files:**
- Modify: `local_services/avatar_memory.py` (add a module-level function)
- Test: `archive/_memory_gating_test.py`

**Interfaces:**
- Produces: `needs_rewrite(text: str, profile: dict) -> bool` — True when the utterance is context-dependent (follow-up markers) or names no location while a `default_city` is known.

- [ ] **Step 1: Write the failing test**

```python
# archive/_memory_gating_test.py
"""needs_rewrite fires on follow-ups / location-less asks, skips self-contained ones.
Run: python -m archive._memory_gating_test"""
from local_services.avatar_memory import needs_rewrite


def main() -> None:
    prof_city = {"default_city": "台北市"}
    prof_none = {"default_city": ""}

    # follow-up markers -> rewrite
    assert needs_rewrite("那台中呢？", prof_city) is True   # "那台中呢?"
    assert needs_rewrite("後天呢？", prof_city) is True          # "後天呢?"
    # no location named + we know their city -> rewrite (fill it in)
    assert needs_rewrite("明天會下雨嗎？", prof_city) is True  # "明天會下雨嗎?"
    # self-contained (names a city) + nothing to add -> skip
    assert needs_rewrite("明天台南會下雨嗎？", prof_none) is False  # names 台南
    # location-less but no profile city to inject -> skip (nothing to add)
    assert needs_rewrite("明天會下雨嗎？", prof_none) is False
    print("PASS _memory_gating_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._memory_gating_test`
Expected: FAIL — `ImportError: cannot import name 'needs_rewrite'`

- [ ] **Step 3: Add `needs_rewrite` to `avatar_memory.py`**

Add near the top of `avatar_memory.py` (after `_DEFAULT_PROFILE`):

```python
# Follow-up / ellipsis markers that signal the utterance leans on prior context.
_FOLLOWUP_MARKERS = (
    "那",      # 那
    "呢",      # 呢
    "還有",  # 還有
    "同樣",  # 同樣
    "一樣",  # 一樣
    "剛",      # 剛
    "這個",  # 這個
    "那個",  # 那個
    "它",      # 它
)
# Taiwan city/county tokens; if none appears, the ask has no explicit location.
_TW_LOCATIONS = (
    "台北", "新北", "桃園", "台中", "台南",
    "高雄", "基隆", "新竹", "苗栗", "彰化",
    "南投", "雲林", "嘉義", "屏東", "宜蘭",
    "花蓮", "台東", "澎湖", "金門", "馬祖",
)


def needs_rewrite(text: str, profile: dict) -> bool:
    """Gate the (latency-costing) rewrite: only rewrite a context-dependent ask.

    True if the utterance has a follow-up marker, OR it names no location while we
    know the user's default_city (so the rewrite can fill it in). Otherwise skip --
    a self-contained query goes straight to the chain.
    """
    if any(mark in text for mark in _FOLLOWUP_MARKERS):
        return True
    has_location = any(loc in text for loc in _TW_LOCATIONS)
    if not has_location and profile.get("default_city"):
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m archive._memory_gating_test`
Expected: `PASS _memory_gating_test`

- [ ] **Step 5: Commit**

```bash
git add local_services/avatar_memory.py archive/_memory_gating_test.py
git commit -m "feat(memory): pure rewrite-gating heuristic"
```

---

### Task 5: Memory LLM ops — build_query (rewrite) + distill

**Files:**
- Modify: `local_services/avatar_memory.py` (add async methods + httpx client)
- Test: `archive/_memory_rewrite_test.py` (LIVE — needs Ollama + `qwen2.5:3b-cpu`)

**Interfaces:**
- Consumes: `needs_rewrite()`, `recall()`, `self.profile`, `self.summary`, `self.session` (Tasks 3–4).
- Produces:
  - `async build_query(self, raw: str) -> str` — gated rewrite; returns a self-contained zh query, or `raw` on skip/failure.
  - `async distill_and_save(self) -> None` — update profile + summary from the session via qwen, persist, reset session.
  - `async aclose(self) -> None` — close the httpx client.

- [ ] **Step 1: Write the failing (live) test**

```python
# archive/_memory_rewrite_test.py
"""LIVE: build_query rewrites a follow-up via local qwen, and degrades to raw when
the LLM is unreachable. Requires Ollama running with qwen2.5:3b-cpu.
Run: python -m archive._memory_rewrite_test"""
import asyncio
import tempfile

from local_services.avatar_memory import MemoryStore


async def run() -> None:
    d = tempfile.mkdtemp()
    m = MemoryStore(
        base_dir=d, enabled=True,
        llm_url="http://localhost:11434/v1", llm_model="qwen2.5:3b-cpu", gated=True,
    )
    m.profile["default_city"] = "台北市"  # Taipei
    m.record_turn("明天台北市會下雨嗎？", "會的")  # prior: rain tomorrow Taipei
    # follow-up should be rewritten into a self-contained zh question mentioning 台中
    out = await m.build_query("那台中呢？")  # "那台中呢?"
    assert "台中" in out, f"expected 台中 in rewrite, got: {len(out)} chars"
    assert "天" in out or "雨" in out, "rewrite lost the weather topic"
    await m.aclose()

    # degradation: bad URL -> returns raw, never raises
    bad = MemoryStore(base_dir=tempfile.mkdtemp(), enabled=True,
                      llm_url="http://127.0.0.1:1/v1", llm_model="x", gated=False)
    raw = "明天會下雨嗎？"
    assert await bad.build_query(raw) == raw
    await bad.aclose()
    print("PASS _memory_rewrite_test")


if __name__ == "__main__":
    asyncio.run(run())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._memory_rewrite_test`
Expected: FAIL — `AttributeError: 'MemoryStore' object has no attribute 'build_query'`

- [ ] **Step 3: Add the LLM ops to `avatar_memory.py`**

Add `import httpx` and `import re` at the top, then these methods on `MemoryStore`:

```python
    # ---- local-LLM client (Ollama, OpenAI-compatible) ----
    def _client(self) -> "httpx.AsyncClient":
        if getattr(self, "_http", None) is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0))
        return self._http

    async def aclose(self) -> None:
        if getattr(self, "_http", None) is not None:
            await self._http.aclose()
            self._http = None

    async def _chat(self, prompt: str, max_tokens: int) -> str:
        """One non-streaming completion from the local model. Raises on failure."""
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        r = await self._client().post(self._llm_url + "/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    # ---- context engineering: rewrite the utterance into a self-contained query ----
    async def build_query(self, raw: str) -> str:
        if not self.enabled or not self._llm_url or not raw:
            return raw
        if self._gated and not needs_rewrite(raw, self.profile):
            return raw
        last = self.session[-1]["user"] if self.session else "（無）"  # "(none)"
        prompt = (
            "任務：把「目前問題」改寫成一句完整、可獨立查詢的繁體中文天氣問題。"
            "利用記憶與上一輪補上缺少的地點或時間。只輸出改寫後的問題，不要解釋。\n\n"
            f"記憶：{self.recall() or '（無）'}\n"
            f"上一輪：{last}\n"
            f"目前問題：{raw}\n改寫："
        )
        try:
            out = await self._chat(prompt, max_tokens=48)
        except Exception as e:  # noqa: BLE001 -- memory must never break a turn
            logger.warning(f"rewrite failed ({type(e).__name__}); using raw utterance")
            return raw
        out = out.splitlines()[0].strip().strip('"「」') if out else ""
        return out or raw

    # ---- harness: distill the conversation into durable memory ----
    async def distill_and_save(self) -> None:
        if not self.enabled or not self._llm_url or not self.session:
            return
        convo = "\n".join(
            f"使用者：{t['user']}\n助理：{t['bot']}" for t in self.session
        )
        prompt = (
            "你是記憶整理助手。讀下面的對話，更新使用者記憶。"
            "只輸出一個 JSON，欄位：name、default_city、preferences(陣列)、summary(繁體中文一段話)。"
            "不確定的欄位保留舊值。\n\n"
            f"舊資料：name={self.profile.get('name')}, default_city={self.profile.get('default_city')}, "
            f"preferences={self.profile.get('preferences')}\n舊摘要：{self.summary or '（無）'}\n\n"
            f"對話：\n{convo}\n\nJSON："
        )
        try:
            out = await self._chat(prompt, max_tokens=400)
            data = _extract_json(out)
            if data:
                self.profile["name"] = data.get("name") or self.profile.get("name", "")
                self.profile["default_city"] = data.get("default_city") or self.profile.get("default_city", "")
                if isinstance(data.get("preferences"), list):
                    self.profile["preferences"] = data["preferences"]
                if data.get("summary"):
                    self.summary = str(data["summary"]).strip()
                self._save_profile()
                self._save_summary()
                logger.info("memory distilled + saved")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"distill failed ({type(e).__name__}); memory unchanged")
        finally:
            self.reset_session()
            await self.aclose()
```

Also add this module-level helper at the bottom of `avatar_memory.py`:

```python
def _extract_json(text: str) -> Optional[dict]:
    """First JSON object in a model reply (handles ```json fences / prose around it)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
```

- [ ] **Step 4: Run the live test to verify it passes**

Pre-req: Ollama running with the model — `ollama list | grep qwen2.5:3b-cpu` (create per Task 9 setup if missing).
Run: `python -m archive._memory_rewrite_test`
Expected: `PASS _memory_rewrite_test` (the rewrite contains 台中 + a weather token; the bad-URL path returns raw)

- [ ] **Step 5: Commit**

```bash
git add local_services/avatar_memory.py archive/_memory_rewrite_test.py
git commit -m "feat(memory): local-qwen build_query (rewrite) + distill_and_save"
```

---

### Task 6: WeatherChainLLMService (the pipeline node)

**Files:**
- Modify: `local_services/weather_chain_llm.py` (add the service class)
- Test: verification via `python -m scripts.preflight` (import-clean) + the live run in Task 8. Add a tiny non-network unit for `_last_user_text` in `archive/_weather_extract_test.py`.

**Interfaces:**
- Consumes: `extract_sse_text()` (Task 2); an optional `MemoryStore` (Tasks 3–5).
- Produces: `class WeatherChainLLMService(LLMService)` with `__init__(self, *, url, model, memory=None, **kwargs)`; static `_last_user_text(context) -> str`.

- [ ] **Step 1: Write the failing test (pure extractor)**

```python
# archive/_weather_extract_test.py
"""_last_user_text pulls the latest user utterance from a universal LLM context
(content as str or as a list of text parts). No network.
Run: python -m archive._weather_extract_test"""
from local_services.weather_chain_llm import WeatherChainLLMService as W


class _Ctx:
    def __init__(self, messages):
        self._m = messages
    def get_messages(self):
        return self._m


def main() -> None:
    # string content
    c1 = _Ctx([{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])
    assert W._last_user_text(c1) == "hi"
    # list-of-parts content, picks the LAST user msg
    c2 = _Ctx([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ])
    assert W._last_user_text(c2) == "second"
    # no user msg
    assert W._last_user_text(_Ctx([{"role": "system", "content": "x"}])) == ""
    print("PASS _weather_extract_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._weather_extract_test`
Expected: FAIL — `AttributeError: type object 'WeatherChainLLMService' has no attribute '_last_user_text'` (the class doesn't exist yet)

- [ ] **Step 3: Add the service class to `weather_chain_llm.py`**

Add imports at the top of the file and the class after `extract_sse_text`:

```python
from typing import AsyncIterator

import httpx
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


class WeatherChainLLMService(LLMService):
    """Streams answers from the remote LangServe weather chain, optionally rewriting
    the user's utterance through a local MemoryStore first (context engineering)."""

    def __init__(self, *, url: str, model: str, memory=None, **kwargs):
        super().__init__(**kwargs)
        self._url = url.rstrip("/") + "/stream"
        self._model = model
        self._memory = memory
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    @staticmethod
    def _last_user_text(context) -> str:
        for msg in reversed(context.get_messages()):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role != "user":
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return ""
        return ""

    async def _stream_chain(self, query: str) -> AsyncIterator[str]:
        payload = {"input": {"query": query, "model": self._model}}
        async with self._client.stream("POST", self._url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                piece = extract_sse_text(line[5:])
                if piece:
                    yield piece

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        raw = self._last_user_text(frame.context)
        query = raw
        if self._memory is not None and raw:
            try:
                query = await self._memory.build_query(raw)
            except Exception as e:  # noqa: BLE001 -- memory must never break a turn
                logger.warning(f"build_query failed ({type(e).__name__}); using raw")
                query = raw

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        answer = ""
        try:
            async for piece in self._stream_chain(query):
                answer += piece
                await self.push_frame(LLMTextFrame(piece))
        except httpx.TimeoutException as e:
            await self.push_error(error_msg="weather chain timeout", exception=e)
            answer = "抱歉，天氣服務反應太慢。"  # timeout fallback
            await self.push_frame(LLMTextFrame(answer))
        except Exception as e:  # noqa: BLE001
            await self.push_error(error_msg=f"weather chain error: {type(e).__name__}", exception=e)
            answer = "抱歉，天氣服務暫時連線不上。"  # connect fallback
            await self.push_frame(LLMTextFrame(answer))
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

        if self._memory is not None and raw:
            try:
                self._memory.record_turn(raw, answer)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"record_turn failed ({type(e).__name__})")
```

- [ ] **Step 4: Run tests + preflight to verify**

Run: `python -m archive._weather_extract_test`
Expected: `PASS _weather_extract_test`
Run: `python -m scripts.preflight`
Expected: preflight reports OK (all imports resolve; no network/keys needed)

- [ ] **Step 5: Commit**

```bash
git add local_services/weather_chain_llm.py archive/_weather_extract_test.py
git commit -m "feat(weather): WeatherChainLLMService streaming node + memory hook"
```

---

### Task 7: Factory branch in `build_llm`

**Files:**
- Modify: `pipeline/stages/llm.py`
- Test: `archive/_llm_factory_test.py`

**Interfaces:**
- Consumes: `config.llm_provider`, `WeatherChainLLMService` (Task 6), the optional `MemoryStore`.
- Produces: `build_llm(cfg, memory=None)` returns `WeatherChainLLMService` when `cfg.llm_provider == "weather_chain"`, else the OpenAI service.

- [ ] **Step 1: Write the failing test**

```python
# archive/_llm_factory_test.py
"""build_llm returns the weather service under LLM_PROVIDER=weather_chain, else OpenAI.
Run: python -m archive._llm_factory_test"""
import dataclasses

from pipeline.config import config
from pipeline.stages.llm import build_llm
from local_services.weather_chain_llm import WeatherChainLLMService


def main() -> None:
    w = build_llm(dataclasses.replace(config, llm_provider="weather_chain"))
    assert isinstance(w, WeatherChainLLMService), type(w)

    o = build_llm(dataclasses.replace(config, llm_provider="openrouter",
                                      openrouter_api_key="sk-test"))
    assert o.__class__.__name__ == "OpenAILLMService", o.__class__.__name__
    print("PASS _llm_factory_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m archive._llm_factory_test`
Expected: FAIL — `TypeError: build_llm() ... unexpected` or an assertion error (current `build_llm` ignores provider and always returns OpenAI)

- [ ] **Step 3: Rewrite `pipeline/stages/llm.py`**

```python
"""LLM stage factory. Two deliberate, single-provider branches (a fallback switch,
not multi-provider branching), chosen by LLM_PROVIDER:
  weather_chain -> WeatherChainLLMService (dedicated zh weather bot, remote LangServe)
  openrouter    -> OpenAILLMService against OpenRouter (general chat)
Tokens stream so the first sentence reaches TTS before the full answer is done."""
from __future__ import annotations

from pipeline.config import Config


def build_llm(cfg: Config, memory=None):
    if cfg.llm_provider == "weather_chain":
        # Local import keeps this branch off the OpenRouter path (and preflight clean).
        from local_services.weather_chain_llm import WeatherChainLLMService

        return WeatherChainLLMService(
            url=cfg.weather_chain_url,
            model=cfg.weather_chain_model,
            memory=memory,
        )

    from pipecat.services.openai.llm import OpenAILLMService

    # `model=` is deprecated; the model now lives in the `settings=` object.
    return OpenAILLMService(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        settings=OpenAILLMService.Settings(model=cfg.openrouter_model),
    )
```

- [ ] **Step 4: Run test + preflight to verify**

Run: `python -m archive._llm_factory_test`
Expected: `PASS _llm_factory_test`
Run: `python -m scripts.preflight`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/llm.py archive/_llm_factory_test.py
git commit -m "feat(llm): branch build_llm on LLM_PROVIDER (weather_chain | openrouter)"
```

---

### Task 8: Wire memory + provider into `main.py`

**Files:**
- Modify: `pipeline/main.py` (~lines 45–128: log line, build_llm call, warmup guard, greeting, disconnect)
- Test: verification via `python -m scripts.preflight` + a live run (no unit test — this is pipeline assembly, matching the repo's verify-via-harness practice).

**Interfaces:**
- Consumes: `build_llm(cfg, memory)` (Task 7), `MemoryStore` (Tasks 3–5), `config` knobs (Task 1).

- [ ] **Step 1: Add the MemoryStore import + creation**

At the top of `pipeline/main.py` with the other `from pipeline.stages import ...` imports, add:

```python
from local_services.avatar_memory import MemoryStore
```

Replace the `llm = build_llm(config)` line (~line 51) with:

```python
    # Memory harness: only for the weather bot, only when enabled. Wrapped around the
    # stateless chain (the chain can't hold memory); rewrites the query + distills the
    # conversation. Local qwen on CPU, so the GPU stays free for the avatar.
    memory = None
    if config.llm_provider == "weather_chain" and config.avatar_memory:
        memory = MemoryStore(
            base_dir=config.avatar_memory_dir,
            llm_url=config.memory_llm_url,
            llm_model=config.memory_llm_model,
            gated=config.memory_llm_gated,
            enabled=True,
        )
        logger.info(f"Avatar memory ON (model={config.memory_llm_model}, gated={config.memory_llm_gated}).")
    llm = build_llm(config, memory)
```

- [ ] **Step 2: Update the startup log line**

Replace the `logger.info(f"Pipeline: Deepgram STT -> OpenRouter LLM -> ...")` block (~line 45) with a provider-aware line:

```python
    _llm_label = "WeatherChain" if config.llm_provider == "weather_chain" else "OpenRouter"
    logger.info(
        f"Pipeline: Deepgram STT -> {_llm_label} LLM -> CosyVoice TTS -> MuseTalk avatar "
        f"(lang={config.language})"
    )
```

- [ ] **Step 3: Guard `_warmup_llm` (OpenAI-only) so it no-ops for the weather bot**

Replace the body of `_warmup_llm` (~lines 96–110) with a provider guard at the top:

```python
    async def _warmup_llm():
        # The chat.completions warmup is OpenAI-specific; the weather chain has no
        # cheap warmup ping, so skip it there (it would crash on the custom service).
        if config.llm_provider == "weather_chain":
            return
        model = getattr(getattr(llm, "_settings", None), "model", None)
        try:
            await llm._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            )
            logger.info("LLM connection pre-warmed.")
        except Exception as e:  # noqa: BLE001 -- best-effort only
            logger.info(f"LLM warmup skipped: {e}")
```

- [ ] **Step 4: Personalize the greeting from memory (Mandarin path)**

In `_on_connected` (~lines 116–123), replace the greeting selection so the Mandarin branch appends the memory hint when present:

```python
        if config.is_thai:
            greeting = "สวัสดีค่ะ พร้อมแล้วค่ะ พูดได้เลย"
        elif config.is_mandarin:
            greeting = "嘿，我準備好了，請說。"  # "Hi, I'm ready, go ahead."
            if memory is not None:
                hint = memory.greeting_hint()
                if hint:
                    greeting = "嘿，歡迎回來！" + hint  # "Hi, welcome back! " + hint
        else:
            greeting = "Hi, I'm ready - go ahead."
```

- [ ] **Step 5: Distill on disconnect**

In `_on_disconnected` (~lines 125–128), distill memory before cancelling the task:

```python
    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(transport, client):
        logger.info(f"Client disconnected. TTFO summary: {meter.summary()}")
        if memory is not None:
            try:
                await memory.distill_and_save()  # grow the human's memory after the chat
            except Exception as e:  # noqa: BLE001
                logger.warning(f"memory distill skipped ({type(e).__name__})")
        await task.cancel()
```

- [ ] **Step 6: Verify imports + a dry run**

Run: `python -m scripts.preflight`
Expected: OK (imports resolve)
Run (smoke, no client needed — just that it boots and logs the provider line, then Ctrl-C): `python -m pipeline.main`
Expected: logs `Pipeline: Deepgram STT -> WeatherChain LLM -> ...` and `Avatar memory ON (...)` when `.env` has `LLM_PROVIDER=weather_chain`.

- [ ] **Step 7: Commit**

```bash
git add pipeline/main.py
git commit -m "feat(pipeline): wire memory store + provider into main (warmup guard, greeting, distill)"
```

---

### Task 9: Probe script, .env docs, ignore the state dir, Ollama setup

**Files:**
- Create: `scripts/probe_weather_chain.py`
- Modify: `.env.example` (create if absent), `.gitignore`, `WORKFLOW.md` (§8 if present)
- Setup: the `qwen2.5:3b-cpu` Ollama model (idempotent)

**Interfaces:** none (tooling/docs).

- [ ] **Step 1: Create the SSE probe**

```python
# scripts/probe_weather_chain.py
"""Standalone probe for the weather chain's /stream SSE shape -- run the instant the
NCU server is reachable to confirm extract_sse_text() parses it (the chunk shape was
an assumption at build time; the server was down).
Run: python -m scripts.probe_weather_chain ["zh question"]"""
import sys

import httpx

from pipeline.config import config
from local_services.weather_chain_llm import extract_sse_text


def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "明天台北市有下雨嗎？"
    url = config.weather_chain_url.rstrip("/") + "/stream"
    payload = {"input": {"query": q, "model": config.weather_chain_model}}
    print(f"POST {url}  model={config.weather_chain_model}")
    text = ""
    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # show raw line lengths + parsed piece length (avoid printing zh to cp1252)
                parsed = extract_sse_text(line[5:]) if line.startswith("data:") else None
                print(f"  raw[{len(line)}] parsed={'<%d>' % len(parsed) if parsed else None}")
                if parsed:
                    text += parsed
    except Exception as e:  # noqa: BLE001
        print(f"PROBE ERROR: {type(e).__name__}: {e}")
        return
    print(f"TOTAL parsed chars: {len(text)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Ignore the runtime state dir**

Append to `.gitignore`:

```
# Avatar memory (per-user runtime state)
state/
```

- [ ] **Step 3: Document the knobs**

Add to `.env.example` (create the file if it doesn't exist), and mirror under `WORKFLOW.md` §8 if that section exists:

```dotenv
# --- LLM provider (fallback switch) ---
LLM_PROVIDER=weather_chain          # weather_chain | openrouter
WEATHER_CHAIN_URL=http://140.115.54.87:8000/chain/resWeatherChain
WEATHER_CHAIN_MODEL=gemma3:27b
LANGUAGE=zh                          # the weather chain requires Chinese in/out

# --- Avatar memory harness (fully local, CPU) ---
AVATAR_MEMORY=1                      # 0 = disable memory
AVATAR_MEMORY_DIR=state/avatar_memory
MEMORY_LLM_URL=http://localhost:11434/v1
MEMORY_LLM_MODEL=qwen2.5:3b-cpu      # CPU-pinned; set qwen2.5:3b to use the GPU
MEMORY_LLM_GATED=1                   # 0 = rewrite every turn
```

- [ ] **Step 4: Ensure the CPU model exists (idempotent setup)**

Run:
```bash
ollama list | grep -q "qwen2.5:3b-cpu" || \
  (printf 'FROM qwen2.5:3b\nPARAMETER num_gpu 0\n' > /tmp/qwen-cpu.Modelfile && \
   ollama create qwen2.5:3b-cpu -f /tmp/qwen-cpu.Modelfile)
```
Expected: model present (`ollama ps` shows `100% CPU` when it serves).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_weather_chain.py .gitignore .env.example
git commit -m "chore(weather): SSE probe, .env docs, ignore state dir"
```

---

## Self-Review

**Spec coverage:**
- Req 1 (replace LLM with weather chain) → Tasks 2, 6, 7, 8. ✓
- Req 2 (Chinese) → `LANGUAGE=zh` doc (Task 9) + zh prompts/greeting (Tasks 5, 8). ✓
- Req 3 components 6–9 (MemoryStore, gated rewrite, distill, config+degradation) → Tasks 1, 3, 4, 5, 8. ✓
- Fully-local CPU qwen → Tasks 1 (defaults), 5, 9 (model). ✓
- Probe + SSE-assumption hedge → Tasks 2, 9. ✓
- Graceful degradation → try/except in Tasks 5, 6, 8; disabled-store inert (Task 3). ✓
- Revert flags (`LLM_PROVIDER`, `AVATAR_MEMORY`) → Tasks 7, 8. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; error handling is concrete (specific fallbacks + ASCII-only logging). ✓

**Type consistency:** `build_query`/`record_turn`/`recall`/`greeting_hint`/`distill_and_save`/`needs_rewrite`/`extract_sse_text`/`_last_user_text`/`build_llm(cfg, memory)` names + signatures match across Tasks 1–8. `MemoryStore.__init__` keyword args (`base_dir, llm_url, llm_model, gated, enabled`) are used identically in Tasks 3, 5, 8. ✓

**Note for the implementer:** the live tests (Task 5, the Task 8 smoke, the Task 9 probe) need Ollama up; the weather-chain live turn additionally needs the NCU server reachable (was down at planning time — use the probe first). All pure-logic tests (Tasks 1–4, 6 extractor, 7) run with no network.
