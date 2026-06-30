"""
Phase 6 — Python client. Call the TTS service from another machine on the LAN.

  python client_example.py "今天台北天氣晴朗" --host 192.168.1.50 --out reply.wav
"""
import argparse
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="reply.wav")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/tts"
    r = requests.post(url, json={"text": args.text, "speed": args.speed}, timeout=120)
    r.raise_for_status()
    with open(args.out, "wb") as f:
        f.write(r.content)
    print(f"[OK] {args.out}  "
          f"gen={r.headers.get('X-Generation-Seconds')}s  "
          f"audio={r.headers.get('X-Audio-Seconds')}s  "
          f"RTF={r.headers.get('X-RTF')}")


if __name__ == "__main__":
    main()
