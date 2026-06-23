"""Standalone probe for the weather chain's /stream SSE shape -- run the instant the
NCU server is reachable to confirm extract_sse_text() parses it (the chunk shape was
an assumption at build time; the server was down).
Run: python -m scripts.probe_weather_chain ["zh question"]"""
import sys

import httpx

from pipeline.config import config
from local_services.weather_chain_llm import extract_sse_text


def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "明天台北市有下雨嗎？"
    url = config.weather_chain_url.rstrip("/") + "/stream"
    payload = {"input": {"query": q, "model": config.weather_chain_model}}
    print(f"POST {url}  model={config.weather_chain_model}")
    text = ""
    try:
        with httpx.stream("POST", url, json=payload, timeout=60.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # show raw line lengths + parsed piece length (avoid printing zh to cp1252)
                parsed = extract_sse_text(line[5:]) if line.startswith("data:") else None
                tag = ("<%d>" % len(parsed)) if parsed else None
                print(f"  raw[{len(line)}] parsed={tag}")
                if parsed:
                    text += parsed
    except Exception as e:  # noqa: BLE001
        print(f"PROBE ERROR: {type(e).__name__}: {e}")
        return
    print(f"TOTAL parsed chars: {len(text)}")


if __name__ == "__main__":
    main()
