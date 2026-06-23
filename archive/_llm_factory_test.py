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
