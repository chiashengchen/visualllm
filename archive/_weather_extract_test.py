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
