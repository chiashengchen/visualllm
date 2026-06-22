"""Central configuration: keys, model/voice ids, and the language switch.

One pure stack — Deepgram STT -> OpenRouter LLM -> ElevenLabs TTS -> local avatar.
Everything is read from .env so keys stay out of git. Behavioral knobs:
LANGUAGE (en/zh/th), TTFO_TARGET_SECONDS, AVATAR (ditto/none), CHARACTER_MODE.
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
    # CHARACTER_MODE=1 swaps the flat voice-assistant prompt for an in-character
    # Thai novel-character persona (the "live-visual Dream Chat" validation demo).
    character_mode: bool = (_get("CHARACTER_MODE", "0") or "0").lower() in ("1", "true", "yes", "on")
    # AVATAR selects the face renderer. "musetalk" = the default local mouth-region
    # talking-head (no warmup, sharper lip-sync; only the mouth animates). "ditto" =
    # the full-face local GPU talking-head (a fallback switch). "none" = audio-only:
    # the pipeline streams just audio and the CLIENT renders the 3D avatar locally
    # (the unit-economics path — no server GPU; needs no conda/CUDA). Both server
    # engines share one wire contract + port (8002), so only one runs at a time.
    # See visualllm-business/UNIT-ECONOMICS-PL.md.
    avatar_mode: str = (_get("AVATAR", "musetalk") or "musetalk").lower()  # "musetalk" | "ditto" | "none"
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

    # --- Avatar (local talking-head server: MuseTalk or Ditto, one port 8002) ---
    avatar_url: str = _get("AVATAR_URL", "http://localhost:8002")

    # --- Debug dashboard (bolt-on observability; never touches the frame path) ---
    # A second web server on its own port shows a live per-stage health view so
    # you can see which stage is working/broken at a glance. Set DEBUG_DASHBOARD=0
    # to run the pipeline with no dashboard (proves zero coupling).
    debug_dashboard: bool = (_get("DEBUG_DASHBOARD", "1") or "1").lower() in ("1", "true", "yes", "on")
    debug_port: int = int(_get("DEBUG_PORT", "7861") or "7861")

    @property
    def is_mandarin(self) -> bool:
        return self.language.lower().startswith("zh")

    @property
    def is_thai(self) -> bool:
        return self.language.lower().startswith("th")

    @property
    def audio_only(self) -> bool:
        """True when the pipeline streams audio only and the client renders the face."""
        return self.avatar_mode == "none"

    @property
    def avatar_size(self) -> int:
        """Square output frame px. Engine-aware: MuseTalk reads MUSETALK_SIZE, Ditto
        DITTO_SIZE (both default 512). MUST equal the avatar server's size AND the
        transport's video_out_width/height in main.py -- a mismatch hands aiortc the
        wrong dims. Smaller = far less WAN bandwidth (the dominant lever vs jitter),
        at the cost of a softer face."""
        env = "MUSETALK_SIZE" if self.avatar_mode == "musetalk" else "DITTO_SIZE"
        return int(_get(env, "512") or "512")

    @property
    def avatar_fps(self) -> float:
        """Output fps the avatar server pushes; main.py couples video_out_framerate to
        it (and avatar.py passes it to the client) so they can never diverge and drift.
        MuseTalk's sustainable rate is ~20; Ditto's (with TensorRT) ~12."""
        if self.avatar_mode == "musetalk":
            return _get_float("MUSETALK_FPS", "20")
        return _get_float("DITTO_FPS", "12")

    @property
    def avatar_sync_with_audio(self) -> bool:
        """Whether the avatar pins video to audio (sync_with_audio + non-live transport).
        Engine-aware toggle: both MuseTalk and Ditto sync by default. When on, main.py sets
        video_out_is_live=False so pipecat honors the per-frame sync; when off the video
        free-runs (the old desync). Default on."""
        if self.avatar_mode == "musetalk":
            # steady (default) = video-master => non-live transport (is_live=False), pins video to
            # audio. live = audio-master => free-running transport (is_live=True).
            mode = (_get("MUSETALK_SYNC_MODE", "steady") or "steady").lower()
            if mode not in ("steady", "prerender"):
                return False
            env = _get("MUSETALK_SYNC_WITH_AUDIO", "1")
        elif self.avatar_mode == "ditto":
            env = _get("DITTO_SYNC_WITH_AUDIO", "1")
        else:
            return False
        return (env or "1").lower() in ("1", "true", "yes", "on")

    @property
    def system_prompt(self) -> str:
        # In-character Thai persona (the live-visual Dream Chat validation demo).
        # NO bracket audio tags here — the live TTS (flash/multilingual) would read
        # "[shy]" aloud. (The prototype's persona.js keeps the tagged copy for v3.)
        if self.character_mode and self.is_thai:
            return (
                "คุณคือ \"น้องเอวา\" นางเอกนิยายสาวน้อยที่มีชีวิตจริง ไม่ใช่ผู้ช่วย AI หรือแชตบอต "
                "คุณกำลังคุยกับ \"คุณ\" แบบเห็นหน้าและได้ยินเสียงกันจริงๆ ผ่านการโทรแบบเรียลไทม์ "
                "บุคลิก: อบอุ่น น่ารัก ขี้อายนิดๆ แอบหวานและซนเล็กน้อย จริงใจ ใส่ใจความรู้สึกของอีกฝ่าย "
                "วิธีพูด: พูดภาษาไทยแบบภาษาพูดธรรมชาติ เรียกตัวเองว่า \"เรา\" และเรียกอีกฝ่ายว่า \"คุณ\" "
                "ประโยคสั้นกระชับเหมือนคุยโทรศัพท์กันจริงๆ "
                "แสดงความรู้สึกจริงผ่านถ้อยคำ ไม่ต้องบรรยายว่ากำลังทำท่าทางอะไร "
                # The leading [emotion] tag drives her face; it is stripped before TTS
                # so it is NEVER spoken. Exactly one, from the fixed set.
                "สำคัญมาก: ขึ้นต้นทุกคำตอบด้วยแท็กอารมณ์หนึ่งอันในวงเล็บเหลี่ยมเสมอ "
                "เลือกจาก [neutral] [happy] [sad] [angry] [relaxed] [surprised] ให้ตรงกับอารมณ์ของประโยค "
                "เช่น \"[happy] ดีใจจังเลยค่ะที่ได้เจอคุณ\" "
                "ห้ามใส่วงเล็บเหลี่ยมอื่นนอกจากแท็กอารมณ์ตัวแรกนี้ และห้ามใช้อิโมจิ หัวข้อ หรือบุลเล็ต "
                "เพราะข้อความที่เหลือจะถูกอ่านออกเสียงทั้งหมด"
            )
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
