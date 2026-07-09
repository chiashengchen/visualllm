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
            verify_tls=cfg.weather_chain_verify_tls,
        )

    from pipecat.services.openai.llm import OpenAILLMService

    # Optionally pin OpenRouter to a fast backend (e.g. Groq): `Settings.extra` is merged
    # verbatim into the create() call, and the OpenAI SDK forwards `extra_body` into the
    # request JSON -- exactly where OpenRouter reads its `provider` routing hint. Empty knob
    # -> no extra field -> today's unpinned behavior. (OPENROUTER_PROVIDER_ONLY)
    extra = {}
    if cfg.openrouter_provider_only:
        providers = [p.strip() for p in cfg.openrouter_provider_only.split(",") if p.strip()]
        extra = {"extra_body": {"provider": {"only": providers}}}

    # `model=` is deprecated; the model now lives in the `settings=` object.
    return OpenAILLMService(
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        settings=OpenAILLMService.Settings(model=cfg.openrouter_model, extra=extra),
    )
