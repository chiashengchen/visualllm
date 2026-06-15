"""Text-to-speech factory. Streaming with a fast first audio chunk is what feeds
the avatar quickly enough to hit the TTFO target.

Providers:
- elevenlabs      : Flash/Turbo, low first-chunk latency — prototype default.
- cartesia        : Sonic, often the lowest latency cloud TTS (English-strong).
- azure           : excellent zh-TW neural voices + Asia-region (low latency TW).
- cosyvoice_local : CosyVoice2-0.5B, ~150ms first chunk, native zh-TW (Phase 3).
- kokoro_local    : light local English TTS fallback.
"""
from __future__ import annotations

from pipeline.config import Config


def build_tts(cfg: Config):
    provider = cfg.tts_provider.lower()

    if provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        # Model from .env (default flash = low latency; eleven_multilingual_v2 = warmer/slower).
        # voice_settings are the emotion levers: low stability -> more expressive variance.
        return ElevenLabsTTSService(
            api_key=cfg.elevenlabs_api_key,
            voice_id=cfg.elevenlabs_voice_id,
            model=cfg.elevenlabs_model,
            params=ElevenLabsTTSService.InputParams(
                stability=cfg.elevenlabs_stability,
                similarity_boost=0.75,
                style=cfg.elevenlabs_style,
                use_speaker_boost=True,
            ),
        )

    if provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService

        return CartesiaTTSService(
            api_key=cfg.cartesia_api_key,
            voice_id=cfg.cartesia_voice_id,
            model="sonic-2",            # multilingual; sonic-2 handles zh
        )

    if provider == "azure":
        from pipecat.services.azure.tts import AzureTTSService

        return AzureTTSService(
            api_key=cfg.azure_speech_key,
            region=cfg.azure_speech_region,
            voice=cfg.azure_voice,      # zh-TW-HsiaoChenNeural by default in zh mode
        )

    if provider == "cosyvoice_local":
        # Custom streaming wrapper over a local CosyVoice2 server (Phase 3).
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        return CosyVoiceTTSService(base_url=cfg.cosyvoice_url)

    if provider == "f5_thai_local":
        # Self-host SPIKE: local F5-TTS-THAI server (see local_services/f5_thai_server/).
        # Reuses the same model-agnostic HTTP client — the server speaks the same /tts contract.
        # NOTE: F5-TTS is not natively streaming; expect higher TTFO than CosyVoice2. The spike
        # measures whether the open Thai voice clears quality + the ~3s "feels live" bar.
        from local_services.cosyvoice_tts import CosyVoiceTTSService

        return CosyVoiceTTSService(base_url=cfg.f5_thai_url)

    if provider == "kokoro_local":
        from pipecat.services.kokoro.tts import KokoroTTSService

        return KokoroTTSService()

    raise ValueError(f"Unknown TTS_PROVIDER: {cfg.tts_provider}")
