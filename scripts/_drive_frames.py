"""Drive the MuseTalk server with a WAV and report frames-vs-audio + render fps.

Isolates the server's frame<->audio contract with NO CosyVoice, pipeline, or WebRTC in
the way -- the cleanest way to answer "why don't the avatar frames equal the audio?" and
"why does A/V drift on long turns?". Mirrors the real client feed (20ms real-time-paced
16k chunks, speech_start/speech_end) and separates THREE quantities the log conflates:

  * REAL rendered lip frames (server `video_clock`)  -- these DO equal audio_sec*fps (+/-1)
  * DELIVERED frames (bytes on the wire)             -- extra = pump HELD/duplicate frames
                                                         (frozen last frame) that keep the
                                                         WebRTC track alive when render < fps
  * effective RENDER fps (unique frames / wall span)

Findings this produced (2026-07-01, prod config fps=12, SIZE=256; docs/PROBLEMS-AND-FIXES P16):
  - Alone, PyTorch render is ~20fps-capable (gpu 259ms + composite ~120ms per 8-frame seg vs
    the 667ms/seg budget) -> drift is a FIXED +0.36s startup offset at ANY length.
  - Under GPU contention (run scripts/_gpu_contention_hog.py alongside) the render drops below
    12fps and drift becomes LENGTH-SCALING: PyTorch +0.37s(2.9s)/+1.35s(5.5s)/+3.94s(13.6s).
  - With MUSETALK_TRT=1 the render holds ~12fps under the SAME contention -> drift stays flat
    at +0.36s (held frames 50->4). That is the shared-GPU long-turn drift fix.

NOTE ON PACING: uses ABSOLUTE-deadline sleeps, not cumulative asyncio.sleep(0.02) -- on Windows
the ~15ms timer granularity otherwise makes the feed ~40% slower than real-time and FAKES drift
(cost an hour of wrong-root-cause before the profile caught it). Keep the absolute deadline.

Run (start the MuseTalk server first; close any /client tab -- it is single-client):
  python -m scripts._drive_frames output/reply_concise.wav 12          # real-time paced (default)
  python -m scripts._drive_frames output/reply_concise.wav 12 burst    # un-paced (pure render)
"""
import asyncio, json, sys, wave
import numpy as np
import websockets

URL = "ws://localhost:8002/stream"
WAV = sys.argv[1]
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def load_16k(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if sr != 16000:
        n = int(round(len(a) * 16000 / sr))
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16)
    return a


async def main():
    audio16 = load_16k(WAV)
    audio_s = len(audio16) / 16000
    delivered = 0
    held = 0            # byte-identical to previous frame = pump held during render underflow
    clocks = []
    times = []          # wall time of each delivered frame in the video window
    state = {"in_video": False, "ended": False, "prev": None}
    loop = asyncio.get_event_loop()
    async with websockets.connect(URL, max_size=None) as ws:
        async def reader():
            nonlocal delivered, held
            try:
                while True:
                    m = await ws.recv()
                    if isinstance(m, (bytes, bytearray)):
                        if state["in_video"]:
                            delivered += 1
                            times.append(loop.time())
                            b = bytes(m)
                            if state["prev"] is not None and b == state["prev"]:
                                held += 1
                            state["prev"] = b
                    else:
                        evt = json.loads(m)
                        t = evt.get("type")
                        if t == "video_start":
                            state["in_video"] = True
                        elif t == "video_clock":
                            clocks.append(evt.get("frames"))
                        elif t == "video_end":
                            state["in_video"] = False
                            state["ended"] = True
            except Exception:
                pass
        rt = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "config", "fps": FPS}))
        await ws.send(json.dumps({"type": "speech_start"}))
        chunk = int(16000 * 0.02)
        mode = sys.argv[3] if len(sys.argv) > 3 else "paced"   # paced | burst
        t_start = loop.time()
        for n, i in enumerate(range(0, len(audio16), chunk)):
            await ws.send(audio16[i:i + chunk].tobytes())
            if mode != "burst":
                # ABSOLUTE-deadline pacing (not cumulative sleeps) so Windows timer
                # granularity can't make the feed drift slower than real-time.
                deadline = t_start + (n + 1) * (chunk / 16000)
                dt = deadline - loop.time()
                if dt > 0:
                    await asyncio.sleep(dt)
        await ws.send(json.dumps({"type": "speech_end"}))
        for _ in range(600):  # <= 6s wait for drain
            if state["ended"]:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        rt.cancel()

    ideal_round = int(round(audio_s * FPS))
    real = clocks[-1] if clocks else (delivered - held)   # video_clock = true rendered count
    unique = delivered - held
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    render_fps = unique / span if span > 0 else 0.0
    print(f"WAV={WAV}")
    print(f"  audio          = {audio_s:.3f}s   fps={FPS}   audio*fps={audio_s*FPS:.2f} (round {ideal_round})")
    print(f"  REAL rendered  = {real} frames (video_clock)   -> matches audio? {real - ideal_round:+d}")
    print(f"  DELIVERED      = {delivered} frames  ({held} were HELD/duplicate)")
    print(f"  video window   = {delivered/FPS:.3f}s   vs audio {audio_s:.3f}s   DRIFT {delivered/FPS - audio_s:+.3f}s")
    print(f"  render wall    = {span:.3f}s for {unique} unique frames  -> effective RENDER fps = {render_fps:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
