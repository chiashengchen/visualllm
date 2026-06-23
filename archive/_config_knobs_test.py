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
