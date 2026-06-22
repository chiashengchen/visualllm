"""Throwaway: measure the MuseTalk video-startup latency (the 'face lags the voice' gap).

Drives the server like the live client does and times, from speech_start, when the FIRST
frame and the 14th (lead-gate) frame come back -- in two feed modes:
  drip  = real-time paced (baseline)
  burst = first BURST_S of audio sent with no sleep, then paced (the MUSETALK_FEED_LEAD_S fix)

Run in the musetalk env with the server up and NO browser connected:
  E:\\miniconda3\\envs\\musetalk\\python.exe -m scripts._startup_probe output/cosy_en.wav
"""
import asyncio, json, sys, time, wave
import numpy as np
import websockets

URL = "ws://localhost:8002/stream"
FPS = 12
LEAD = 14
BURST_S = 1.2
WAV = sys.argv[1] if len(sys.argv) > 1 else "output/cosy_en.wav"


def load_16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16)
    return a


async def run(audio16, burst):
    ev = {"video_start": None, "first_real": None, "t0": None}
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            seen_vs = False
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, str):
                        if json.loads(m).get("type") == "video_start" and ev["video_start"] is None:
                            ev["video_start"] = time.perf_counter(); seen_vs = True
                    elif seen_vs and ev["first_real"] is None:
                        ev["first_real"] = time.perf_counter()
            except Exception:
                pass
        rt = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "speech_start"}))
        ev["t0"] = time.perf_counter()
        chunk = int(16000 * 0.02)            # 20ms
        fed = 0.0
        for i in range(0, len(audio16), chunk):
            await ws.send(audio16[i:i + chunk].tobytes())
            fed += 0.02
            if not (burst and fed <= BURST_S):
                await asyncio.sleep(0.02)     # real-time pace
        await ws.send(json.dumps({"type": "speech_end"}))
        await asyncio.sleep(3.0)
        rt.cancel()
    t0 = ev["t0"]
    vs = (ev["video_start"] - t0) if ev["video_start"] else float("nan")
    fr = (ev["first_real"] - t0) if ev["first_real"] else float("nan")
    print(f"  {'BURST' if burst else 'DRIP ':5} -> video_start @ {vs:.2f}s, first rendered frame @ {fr:.2f}s")


async def main():
    a = load_16k(WAV)
    print(f"audio = {len(a)/16000:.2f}s  ({WAV})  fps={FPS} lead={LEAD} burst_s={BURST_S}")
    await run(a, burst=False)
    await asyncio.sleep(1.0)
    await run(a, burst=True)


if __name__ == "__main__":
    asyncio.run(main())
