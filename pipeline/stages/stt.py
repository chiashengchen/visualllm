"""Speech-to-text. Default: Deepgram streaming (nova-2), switching model language by
config so one provider serves the English prototype, the zh-TW target, and the Thai
(th) live-character validation. Local fallbacks (fully offline, CPU, ~0 VRAM):
  STT_PROVIDER=sherpa -> sherpa-onnx STREAMING zipformer (bilingual zh-en); drives
    turn-taking from its own ASR endpoint detector, robust to a quiet/attenuated mic.
  STT_PROVIDER=funasr -> SenseVoice-Small SEGMENTED server (needs the energy-VAD to fire).
Deliberate fallback switches, not multi-provider branching."""
from __future__ import annotations

from pipeline.config import Config


def build_stt(cfg: Config):
    if cfg.stt_provider == "sherpa":
        # Local OFFLINE STREAMING (sherpa-onnx, CPU, ~0 VRAM). Drives turn-taking from its
        # own ASR endpoint detector, so it works even when the energy-VAD doesn't fire.
        from local_services.sherpa_stt import SherpaStreamingSTTService

        return SherpaStreamingSTTService(
            model_dir=cfg.sherpa_model_dir,
            to_traditional=cfg.sherpa_traditional,
            endpoint_silence=cfg.sherpa_endpoint_silence,
        )

    if cfg.stt_provider == "funasr":
        # Local OFFLINE SenseVoice-Small on CPU (~0 VRAM). The server returns
        # Traditional (zh-TW) text via OpenCC, so no pipeline-side conversion.
        from local_services.funasr_stt import FunasrSTTService

        return FunasrSTTService(base_url=cfg.funasr_url)

    from pipecat.services.deepgram.stt import DeepgramSTTService

    if cfg.is_thai:
        language = "th"
    elif cfg.is_mandarin:
        language = "zh-TW"
    else:
        language = "en-US"
    # Pipecat moved per-service tuning into a `settings=` object; the old
    # `live_options=LiveOptions(...)` is deprecated and slated for removal.
    return DeepgramSTTService(
        api_key=cfg.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model="nova-2-general",
            language=language,
            smart_format=True,
        ),
    )
