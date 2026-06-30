"""Text-to-speech.

Default: CosyVoice (TTS_PROVIDER=cosyvoice) -- a local CosyVoice2-0.5B streaming
server (female zero-shot voice), no per-token cloud cost; streams first chunk early
enough to feed the avatar within the TTFO target.

Fallbacks: ElevenLabs streaming (flash_v2_5, multilingual cloud, covers zh-TW) and
Deepgram Aura (TTS_PROVIDER=deepgram, reuses the Deepgram key, English-only). These
are deliberate fallback switches, not a return to multi-provider branching.
"""
from __future__ import annotations

from pipeline.config import Config


def build_tts(cfg: Config):
    if cfg.tts_provider == "cosyvoice":
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        # Local streaming server; native 24 kHz (Pipecat resamples to 16 kHz for the
        # avatar). voice="weather" = the server's registered female zero-shot reference.
        return CosyVoiceTTSService(
            base_url=cfg.cosyvoice_url,
            voice=cfg.cosyvoice_voice,
            sample_rate=cfg.cosyvoice_sample_rate,
        )

    if cfg.tts_provider == "moss":
        # MOSS-TTS-Realtime local server speaks the SAME /tts/stream raw-PCM contract as
        # CosyVoice, so we reuse the CosyVoice client pointed at MOSS_URL. The voice is a
        # fixed professional reference pinned server-side (MOSS_REF), so `voice` is
        # informational here. Native 24 kHz; Pipecat resamples to 16 kHz for the avatar.
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        return CosyVoiceTTSService(
            base_url=cfg.moss_url,
            voice="pro",
            sample_rate=cfg.moss_sample_rate,
        )

    if cfg.tts_provider == "deepgram":
        from pipecat.services.deepgram.tts import DeepgramTTSService

        # Reuses DEEPGRAM_API_KEY (same account as STT). Aura outputs linear16; the
        # avatar/transport resample as needed, so no sample_rate pinning required.
        return DeepgramTTSService(
            api_key=cfg.deepgram_api_key,
            settings=DeepgramTTSService.Settings(voice=cfg.deepgram_tts_voice),
        )

    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    # `voice_id=`/`model=` are deprecated; both now live in `settings=`
    # (note the field is `voice`, not `voice_id`).
    return ElevenLabsTTSService(
        api_key=cfg.elevenlabs_api_key,
        settings=ElevenLabsTTSService.Settings(
            voice=cfg.elevenlabs_voice_id,
            model=cfg.elevenlabs_model,
        ),
    )
