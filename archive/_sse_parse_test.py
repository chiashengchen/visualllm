"""extract_sse_text tolerates the LangServe /stream chunk shapes we might see.
Run: python -m archive._sse_parse_test"""
from local_services.weather_chain_llm import extract_sse_text


def main() -> None:
    # bare JSON string chunk (most likely for a StrOutputParser chain)
    assert extract_sse_text('"明天"') == "明天"
    # object with a content field
    assert extract_sse_text('{"content": "台北"}') == "台北"
    # object with an output field
    assert extract_sse_text('{"output": "下雨"}') == "下雨"
    # raw non-JSON text (some chains stream plain text)
    assert extract_sse_text("hello") == "hello"
    # control / empty payloads -> None
    assert extract_sse_text("[DONE]") is None
    assert extract_sse_text("   ") is None
    assert extract_sse_text('{"event": "metadata"}') is None
    print("PASS _sse_parse_test")


if __name__ == "__main__":
    main()
