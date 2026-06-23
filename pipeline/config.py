"""Central configuration: keys, model/voice ids, and the language switch.

One pure stack — Deepgram STT -> OpenRouter LLM -> CosyVoice TTS -> MuseTalk avatar.
Everything is read from .env so keys stay out of git. Behavioral knobs:
LANGUAGE (en/zh/th), TTFO_TARGET_SECONDS, TTS_PROVIDER, and the MUSETALK_* avatar knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val


def _get_float(name: str, default: str) -> float:
    """Parse a numeric env var, falling back to `default` on blank/garbage.

    os.getenv returns "" (not the default) when a key is present-but-empty in
    .env, so a stray `FOO=` would make float("") blow up at import. Fall back.
    """
    raw = _get(name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            import warnings

            warnings.warn(
                f"{name}={raw!r} is not a number; using default {default}.",
                stacklevel=2,
            )
    return float(default)


@dataclass(frozen=True)
class Config:
    # --- language + targets ---
    language: str = _get("LANGUAGE", "en")  # "en" | "zh" | "th"
    ttfo_target_s: float = _get_float("TTFO_TARGET_SECONDS", "8")

    # --- product mode ---
    # ECHO_GUARD=1 mutes the mic while the bot is speaking (half-duplex), so the
    # avatar's own voice leaking into the mic can't trigger a barge-in that cuts the
    # render mid-turn. Trade-off: you can't interrupt the bot while it talks. Set 0
    # to allow genuine barge-in (and rely on headphones/echo cancellation).
    echo_guard: bool = (_get("ECHO_GUARD", "1") or "1").lower() in ("1", "true", "yes", "on")

    # --- STT (Deepgram) ---
    deepgram_api_key: str | None = _get("DEEPGRAM_API_KEY")

    # --- LLM (OpenRouter: one key, any model via OPENROUTER_MODEL) ---
    openrouter_api_key: str | None = _get("OPENROUTER_API_KEY")
    openrouter_base_url: str = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = _get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")

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

    # --- TTS ---
    # CosyVoice (local CosyVoice2-0.5B streaming server) is the default -- a female
    # zero-shot voice, no per-token cloud cost. TTS_PROVIDER=elevenlabs falls back to
    # ElevenLabs flash_v2_5 (multilingual cloud); TTS_PROVIDER=deepgram to Deepgram
    # Aura (reuses DEEPGRAM_API_KEY, English-only). These are deliberate fallback
    # switches, not a return to multi-provider branching.
    tts_provider: str = (_get("TTS_PROVIDER", "cosyvoice") or "cosyvoice").lower()
    # CosyVoice2 local streaming server (local_services/cosyvoice_tts.py client ->
    # the user's cosyvoice-local-tts FastAPI server). Voice "weather" is its registered
    # female Mandarin zero-shot reference; native rate 24 kHz (Pipecat resamples down).
    cosyvoice_url: str = _get("COSYVOICE_URL", "http://localhost:8001")
    cosyvoice_voice: str = _get("COSYVOICE_VOICE", "weather") or "weather"
    cosyvoice_sample_rate: int = int(_get("COSYVOICE_SAMPLE_RATE", "24000") or "24000")
    elevenlabs_api_key: str | None = _get("ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str | None = _get("ELEVENLABS_VOICE_ID")
    # flash_v2_5 is low-latency and multilingual (covers zh-TW); override for a
    # warmer (slower) voice via ELEVENLABS_MODEL=eleven_multilingual_v2.
    elevenlabs_model: str = _get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    # Deepgram Aura voice (used only when TTS_PROVIDER=deepgram). aura-2-* are the
    # newer, more natural English voices.
    deepgram_tts_voice: str = _get("DEEPGRAM_TTS_VOICE", "aura-2-helena-en") or "aura-2-helena-en"

    # --- Avatar (local MuseTalk talking-head server on port 8002) ---
    avatar_url: str = _get("AVATAR_URL", "http://localhost:8002")

    @property
    def is_mandarin(self) -> bool:
        return self.language.lower().startswith("zh")

    @property
    def is_thai(self) -> bool:
        return self.language.lower().startswith("th")

    @property
    def avatar_size(self) -> int:
        """Square output frame px (MUSETALK_SIZE, default 512). MUST equal the avatar
        server's size AND the transport's video_out_width/height in main.py -- a
        mismatch hands aiortc the wrong dims. Smaller = far less WAN bandwidth (the
        dominant lever vs jitter), at the cost of a softer face."""
        return int(_get("MUSETALK_SIZE", "512") or "512")

    @property
    def avatar_fps(self) -> float:
        """Output fps the avatar server pushes (MUSETALK_FPS, ~20 sustainable); main.py
        couples video_out_framerate to it (and avatar.py passes it to the client) so
        they can never diverge and drift."""
        return _get_float("MUSETALK_FPS", "20")

    @property
    def avatar_sync_with_audio(self) -> bool:
        """Whether the avatar pins video to audio (sync_with_audio + non-live transport).
        steady (default) = video-master => non-live transport (is_live=False), pins video
        to audio. live = audio-master => free-running transport (is_live=True). When on,
        main.py sets video_out_is_live=False so pipecat honors the per-frame sync."""
        mode = (_get("MUSETALK_SYNC_MODE", "steady") or "steady").lower()
        if mode not in ("steady", "prerender"):
            return False
        return (_get("MUSETALK_SYNC_WITH_AUDIO", "1") or "1").lower() in ("1", "true", "yes", "on")

    @property
    def system_prompt(self) -> str:
        if self.is_thai:
            return (
                "คุณเป็นผู้ช่วยด้วยเสียงที่เป็นมิตรและกระชับ "
                "ตอบเป็นภาษาไทยแบบภาษาพูดที่เป็นธรรมชาติ ประโยคสั้นๆ "
                "ตอบสั้นๆ ไม่เกิน 2-3 ประโยคเสมอ ถ้าเรื่องยาวให้ตอบสั้นๆ ก่อนแล้วถามว่าอยากฟังต่อไหม "
                "ห้ามใช้อิโมจิ บุลเล็ต หรือสัญลักษณ์จัดรูปแบบใดๆ เพราะข้อความจะถูกอ่านออกเสียง"
            )
        if self.is_mandarin:
            return (
                "你是一個友善、簡潔的語音助理。"
                "請用口語化、適合朗讀的方式回答，句子要短，"
                "每次回覆都要簡短，最多 2-3 句；內容很多時先給簡短答案再問是否要繼續，不要長篇大論，"
                "避免使用表情符號、條列符號或特殊格式。"
            )
        return (
            "You are a friendly, concise voice assistant. Answer in a natural, "
            "spoken style. Keep sentences short. Do not use emojis, bullet "
            "points, or any special formatting — your text will be read aloud. "
            "Keep every reply brief — at most 2-3 short sentences. If the topic is "
            "big, give the short answer and offer to say more, rather than monologuing."
        )


config = Config()
