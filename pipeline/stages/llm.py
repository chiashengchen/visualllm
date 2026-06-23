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
