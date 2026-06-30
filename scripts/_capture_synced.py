"""Offline avatar capture with CORRECT A/V sync.

Unlike _capture.py (which keeps every frame the server emits, including the ~14 neutral
LEAD frames before speech and the trailing idle frames, so the muxed video starts on a
neutral frame while the audio is already talking), this keeps ONLY the real rendered
frames the server brackets with `video_start` / `video_end`. Those map 1:1 to the audio,
so muxing the audio from t=0 is in sync.

Run in the musetalk env after the server is up:
  python scripts/_capture_synced.py output/demo_cosy.wav
"""
import asyncio, json, subprocess, sys, wave
import numpy as np
import cv2
import websockets

URL = "ws://localhost:8002/stream"
SIZE = 256
FPS = 20
WAV = sys.argv[1] if len(sys.argv) > 1 else "output/cosy_en.wav"
STEM = WAV.rsplit("/", 1)[-1].rsplit(".", 1)[0]
OUT_SILENT = f"output/{STEM}_synced_silent.mp4"
OUT = f"output/{STEM}_synced.mp4"
FFMPEG = r"E:\miniconda3\envs\tts\Library\bin\ffmpeg.exe"


def load_16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16)
    return a


async def main():
    audio16 = load_16k(WAV)
    frames = []
    # Frame side is auto-detected from the first RGB buffer (the server renders at
    # MUSETALK_SIZE, which may be 256 or 512 depending on its env -- don't hardcode it).
    state = {"in_video": False, "ended": False, "side": None}
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        if state["side"] is None:
                            px = len(m) // 3
                            s = int(round(px ** 0.5))
                            if s * s * 3 == len(m):
                                state["side"] = s
                        # Keep a frame ONLY while we're between video_start and video_end:
                        # those are the real rendered (speaking) frames, 1:1 with the audio.
                        if state["in_video"] and state["side"] and len(m) == state["side"] ** 2 * 3:
                            frames.append(bytes(m))
                    else:
                        try:
                            t = json.loads(m).get("type")
                        except Exception:
                            continue
                        if t == "video_start":
                            state["in_video"] = True
                        elif t == "video_end":
                            state["in_video"] = False
                            state["ended"] = True
            except Exception:
                pass
        rt = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))
        chunk = int(16000 * 0.02)
        for i in range(0, len(audio16), chunk):
            await ws.send(audio16[i:i + chunk].tobytes())
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "speech_end"}))
        # Wait for the server to finish draining real frames (video_end), up to a cap.
        for _ in range(400):  # <= 4s
            if state["ended"]:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        rt.cancel()

    audio_s = len(audio16) / 16000
    vid_s = len(frames) / FPS
    side = state["side"] or SIZE
    print(f"captured {len(frames)} REAL frames ({side}x{side}) = {vid_s:.2f}s video for "
          f"{audio_s:.2f}s audio (drift {vid_s - audio_s:+.2f}s)")
    vw = cv2.VideoWriter(OUT_SILENT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (side, side))
    for fb in frames:
        arr = np.frombuffer(fb, dtype=np.uint8).reshape(side, side, 3)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    vw.release()
    # Mux audio from t=0; -shortest trims the (tiny) tail so A and V end together.
    subprocess.run(
        [FFMPEG, "-y", "-i", OUT_SILENT, "-i", WAV, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", OUT],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
