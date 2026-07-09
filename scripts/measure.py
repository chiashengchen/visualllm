"""Unified measurement harness — ONE command for the whole turn.

Replaces running `_webrtc_probe`, the offline `_capture`, and hand-reading
`logs/pipeline.log` separately. It:

  1. drives a real turn through the LIVE pipeline over WebRTC (plays a wav as the
     mic, records the bot's A+V to an mp4, measures receiver-side quality),
  2. parses the `pipeline.log` delta for THAT turn into a per-stage timeline +
     node-to-node handoffs + TTFO/TTFB (all on the pipeline's single clock),
  3. (optional) drives the MuseTalk server directly for a guaranteed lip-offset
     (the WebRTC mp4 sometimes carries no video track),
  4. writes ONE  output/measure_report.json  AND  docs/measure_data.js
     (`window.MEASURE = {...}`) so docs/workflow-timeline.html auto-refreshes.

Run (pipeline + both servers up, no browser on /client):
    python -m scripts.measure --mic output/q_ai.wav --lead 8 --duration 40 --fps 12
    python -m scripts.measure --offline-capture        # also get a clean lip offset
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay
from dotenv import load_dotenv

# Reuse the probe's wav builder + lip-offset analyser (single source of truth).
from scripts._webrtc_probe import build_mic_wav, lip_offset_from_mp4, wait_ice

ROOT = Path(__file__).resolve().parent.parent
# Read the SAME .env the pipeline uses, so the playout-estimate jitter buffer matches the live
# CLIENT_JITTER_BUFFER_MS (150 here) instead of os.getenv's hardcoded 400 default -- otherwise the
# "Browser jitter + decode + playout" waterfall row is overstated by the config delta (~250ms).
load_dotenv(ROOT / ".env")
LOG = ROOT / "logs" / "pipeline.log"
OFFER_URL = "http://127.0.0.1:7860/api/offer"
MP4 = str(ROOT / "output" / "measure_live.mp4")
JSON_OUT = ROOT / "output" / "measure_report.json"
JS_OUT = ROOT / "docs" / "measure_data.js"

# Per-phase glyphs reused by the HTML; kept here so the JS is fully self-describing.
RING, DOT = "◯", "●"  # 'receives'  /  'emits'

_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \| ")


def _audio_rms(frame):
    """RMS of one decoded aiortc AudioFrame (int16 PCM) -> float."""
    s = frame.to_ndarray()
    if s.size == 0:
        return 0.0
    s = s.astype(np.float64)
    return float(np.sqrt(np.mean(s * s)))


# ----------------------------------------------------------------- WebRTC probe
async def run_probe(mic_wav: str, lead: float, tail: float, duration: float):
    """Connect like a browser, play the mic wav, record + time the bot's A/V."""
    vwall: list[float] = []
    awall: list[tuple[float, float]] = []  # (arrival_epoch, rms) per received audio frame
    mic = build_mic_wav(mic_wav, lead, tail)

    pc = RTCPeerConnection()
    pc.addTrack(MediaPlayer(mic).audio)
    pc.addTransceiver("video", direction="recvonly")
    tracks: dict = {}
    pc.on("track", lambda t: tracks.__setitem__(t.kind, t))

    await pc.setLocalDescription(await pc.createOffer())
    await wait_ice(pc)
    connect_t = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.post(OFFER_URL, json={"sdp": pc.localDescription.sdp,
                                           "type": "offer"}) as r:
            ans = await r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))

    for _ in range(50):
        if "video" in tracks and "audio" in tracks:
            break
        await asyncio.sleep(0.1)

    async def vpump(track):
        while True:
            try:
                await track.recv()
            except Exception:
                return
            vwall.append(time.time())

    async def apump(track):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            awall.append((time.time(), _audio_rms(frame)))

    relay = MediaRelay()
    recorder = MediaRecorder(MP4)
    if "video" in tracks:
        recorder.addTrack(relay.subscribe(tracks["video"]))
        asyncio.ensure_future(vpump(relay.subscribe(tracks["video"])))
    if "audio" in tracks:
        recorder.addTrack(relay.subscribe(tracks["audio"]))
        asyncio.ensure_future(apump(relay.subscribe(tracks["audio"])))
    await recorder.start()
    print(f"  connected (pc_id={ans.get('pc_id')}, tracks={list(tracks)}); capturing {duration}s...")
    await asyncio.sleep(duration)
    await recorder.stop()
    await pc.close()
    return vwall, awall, connect_t


def probe_metrics(vwall, awall, connect_t, fps):
    m = {"video_frames": len(vwall), "audio_packets": len(awall)}
    if len(vwall) >= 5:
        w = np.array(vwall)
        gaps = np.diff(w)
        m.update(
            startup_s=round(w[0] - connect_t, 2),
            recv_fps=round(len(vwall) / (w[-1] - w[0] + 1e-9), 1),
            frame_ms_mean=round(gaps.mean() * 1000, 1),
            frame_ms_p95=round(float(np.percentile(gaps, 95)) * 1000, 1),
            frame_ms_max=round(gaps.max() * 1000, 1),
            freeze_ms=round(gaps.max() * 1000),
        )
    if len(awall) > 2:
        ag = np.diff(np.array([t for t, _ in awall]))
        m["audio_gap_p95_ms"] = round(float(np.percentile(ag, 95)) * 1000, 1)
        m["audio_gap_max_ms"] = round(ag.max() * 1000, 1)
    off, corr, err = lip_offset_from_mp4(MP4, fps)
    if err:
        m["lip_offset"] = None
        m["lip_offset_note"] = err
    else:
        m["lip_offset_ms"] = round(off * 1000)
        m["lip_offset_corr"] = round(corr, 2)
    return m


# ------------------------------------------------------------------- log parse
def _parse_lines():
    out = []
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            mt = _TS.match(ln)
            if mt:
                dt = datetime.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S.%f")
                out.append((dt, ln.rstrip("\n")))
    return out


def parse_turn():
    """Find the most recent completed turn (last [TTFO] line) and pull its anchors."""
    lines = _parse_lines()
    ttfo_idx = [i for i, (_, t) in enumerate(lines) if "[TTFO" in t]
    if not ttfo_idx:
        raise SystemExit("No [TTFO ...] line in pipeline.log — did a turn complete?")
    bi = ttfo_idx[-1]
    ttfo_s = ttfo_pass = None
    mt = re.search(r"\[TTFO (OK |OVER)\] ([\d.]+)s", lines[bi][1])
    if mt:
        ttfo_pass = mt.group(1).strip() == "OK"
        ttfo_s = float(mt.group(2))
    return _build_turn(lines, bi, lines[bi][0], ttfo_s, ttfo_pass)


def parse_browser_turn(target_s=3.0, fresh_s=300.0):
    """Anchor the most recent REAL BROWSER turn for --from-browser.

    Browser turns don't emit a [TTFO line: pipecat's transcription turn-stop path (logged
    'strategy: None') broadcasts no user-stop frame the TtfoMeter arms on, and a noisy real mic
    yields no Silero VAD stop either — so parse_turn()'s [TTFO anchor would lock onto a STALE
    headless turn. Instead we anchor on the real [client-playout] beacon: bot_started is the
    'Bot started speaking' whose audio the beacon reports (the server emit precedes the browser
    onset), and we compute the turn's TTFO ourselves. Returns a turn dict, or None if there is no
    beacon / no matching bot-start (e.g. the browser page is stale and never armed the probe)."""
    lines = _parse_lines()
    # Anchor on the beacon's SERVER-CLOCK arrival time (its log timestamp), NOT the browser's
    # Date.now() in the body: the client can be a DIFFERENT device whose clock is skewed from the
    # server's (breaks the same-box assumption), but the server logs bot-start AND the beacon POST
    # on one clock. server_arrival - bot_started = last mile + the onset->POST round-trip (~tens of
    # ms on a LAN) — a small, honest overestimate that is robust to any client clock offset.
    beacon = None
    for dt, txt in reversed(lines):
        if "[client-playout]" in txt and "audio-onset" in txt:
            beacon = dt.timestamp()  # server-clock instant the onset beacon reached us
            break
    if beacon is None:
        return None
    # Freshness guard: reject a beacon logged minutes ago (a stale prior turn / a crafted test) so it
    # can't fake a number — fall back with the "hard-reload + do one turn" guidance instead.
    if time.time() - beacon > fresh_s:
        return None
    # The 'Bot started speaking' whose audio this beacon reports: the server emit precedes the
    # beacon arrival by the last mile, so take the closest bot-start within a 6s window before it.
    bi = None
    for i in range(len(lines) - 1, -1, -1):
        dt, txt = lines[i]
        if "Bot started speaking" in txt and 0 <= beacon - dt.timestamp() <= 6.0:
            bi = i
            break
    if bi is None:
        return None
    turn = _build_turn(lines, bi, lines[bi][0], None, None, target_s=target_s)
    turn["playout_epoch"] = beacon  # server-clock beacon arrival -> the last-mile source
    return turn


def _build_turn(lines, bi, bot_started_t, ttfo_s, ttfo_pass, target_s=3.0):
    """Shared anchor extraction. bi indexes the turn's 'bot start' line (a [TTFO line for the
    headless path, a 'Bot started speaking' line for the browser path). ttfo_s=None means the
    meter never logged it (browser turn) → we derive it as bot_started - t0."""
    # t0 = the 'User stopped speaking' / 'Generating chat' just before the bot start.
    t0 = None
    question = None
    for dt, txt in lines[:bi][::-1]:
        if t0 is None and ("Generating chat from context" in txt or "User stopped speaking" in txt):
            t0 = dt
        if question is None:
            qm = re.search(r"'role': 'user', 'content': '(.*?)'\}\]", txt)
            if qm:
                question = qm.group(1)
        if t0 and question:
            break
    if t0 is None:
        t0 = bot_started_t

    def off(dt):
        return round((dt - t0).total_seconds(), 3)

    if ttfo_s is None:  # browser turn: derive the TTFO the meter never logged
        ttfo_s = round((bot_started_t - t0).total_seconds(), 2)
        ttfo_pass = ttfo_s <= target_s

    turn = {"t0": t0, "ttfo_s": ttfo_s, "ttfo_pass": ttfo_pass,
            "bot_started": off(bot_started_t), "question": question}

    # Scan the window [t0-3s .. t0+60s] for the per-stage anchors of THIS turn.
    win = [(dt, txt) for dt, txt in lines if -3 <= (dt - t0).total_seconds() <= 60]
    user_started = llm_ttfb = bot_stopped = None
    sentences, tts_ttfb, tts_proc = [], [], []
    for dt, txt in win:
        if "User started speaking" in txt and user_started is None:
            user_started = off(dt)
        if "OpenAILLMService" in txt and "TTFB:" in txt and llm_ttfb is None:
            llm_ttfb = (off(dt), float(re.search(r"TTFB: ([\d.]+)s", txt).group(1)))
        m1 = re.search(r"run_tts:\d+ - CosyVoice TTS \[(.*)\]", txt)
        if m1 and dt >= t0:
            sentences.append((off(dt), m1.group(1)))
        if "CosyVoiceTTSService" in txt and "TTFB:" in txt and dt >= t0:
            tts_ttfb.append((off(dt), float(re.search(r"TTFB: ([\d.]+)s", txt).group(1))))
        if "CosyVoiceTTSService" in txt and "processing time:" in txt and dt >= t0:
            tts_proc.append(off(dt))
        if "Bot stopped speaking based on TTSStoppedFrame" in txt:
            bot_stopped = off(dt)
    turn.update(user_started=user_started, llm_ttfb=llm_ttfb, bot_stopped=bot_stopped,
                sentences=sentences, tts_ttfb=tts_ttfb, tts_proc=tts_proc)
    return turn


def answer_onset_epoch(samples, t0_epoch, guard=0.15, thresh_frac=0.18, run=3):
    """First SUSTAINED energetic audio frame after t0 = the answer reaching the client.

    samples: list of (arrival_epoch, rms). Frames at/after t0_epoch+guard are considered, so the
    greeting (well before t0) and inter-turn silence are skipped; the threshold is a fraction of
    the post-t0 peak, and `run` consecutive frames must clear it so a lone spike doesn't trigger.
    Returns the onset epoch (same clock as the log's t0), or None.
    """
    win = [(t, r) for (t, r) in samples if t >= t0_epoch + guard]
    if len(win) < run:
        return None
    peak = max(r for _, r in win)
    if peak <= 0:
        return None
    thr = thresh_frac * peak
    for i in range(len(win) - run + 1):
        if all(win[i + k][1] >= thr for k in range(run)):
            return win[i][0]
    return None


# Ordered stages of the turn; each row's cost ends at the named anchor. Kept module-level so the
# HTML/JS and the tests share one definition of "the stages".
_WATERFALL_STAGES = [
    ("STT finalize -> LLM", "llm_recv", "assumed"),  # llm_recv is pinned to t0 (pre-warmed), not measured
    ("LLM first token", "llm_ttfb", "log"),
    ("LLM -> TTS (sentence-1 flush)", "tts_recv", "log"),
    ("TTS synth first chunk", "tts_ttfb", "log"),
    ("TTS -> bot-start (steady lead-hold)", "bot_started", "log"),
    ("Transport + encode + network", "client_arrival", "probe"),
    ("Browser jitter + decode + playout", "playout", "browser"),
]


def build_waterfall(anchors, playout_source="est"):
    """Per-stage latency from t0 to the user's ear. anchors: dict of t0-relative offsets (s);
    a None anchor yields an 'unknown' row that does NOT corrupt the running sum (the next known
    stage's delta absorbs the gap, so ok-row deltas always telescope to the last known cum).
    A NEGATIVE anchor (logged before t0) is also 'unknown': a stage can't complete before the
    turn starts, so it is a prior-turn artifact (e.g. VAD split a synthetic mic into two turns),
    never a real -X.XXs latency. Returns ordered rows: {stage, delta, cum, source, status}; the
    final 'total' row carries the end-to-end cum. The last stage's source is `playout_source`.
    """
    rows, prev = [], 0.0
    for label, key, source in _WATERFALL_STAGES:
        if key == "playout":
            source = playout_source
        end = anchors.get(key)
        if end is None or end < 0:
            rows.append(dict(stage=label, delta=None, cum=None, source=source, status="unknown"))
            continue
        rows.append(dict(stage=label, delta=round(end - prev, 3), cum=round(end, 3),
                         source=source, status="ok"))
        prev = end
    total = next((r["cum"] for r in reversed(rows) if r["cum"] is not None), None)
    rows.append(dict(stage="END-TO-END, user hears", delta=None, cum=total,
                     source="", status="total"))
    return rows


def parse_playout_beacon(lines, t0_dt):
    """First real-browser [client-playout] audio-onset after t0. lines: list of (datetime, text)
    (as _parse_lines returns). The beacon body is JSON {"ev":"audio-onset","t":<epoch_ms>}.
    Returns the offset from t0 in seconds (same-box clock), or None."""
    t0e = t0_dt.timestamp()
    for dt, txt in lines:
        if "[client-playout]" in txt and "audio-onset" in txt and dt >= t0_dt:
            m = re.search(r'"t":\s*(\d+(?:\.\d+)?)', txt)
            if m:
                return round(float(m.group(1)) / 1000.0 - t0e, 3)
    return None


# --------------------------------------------------------- assemble timeline JS
def build_events(turn):
    """Turn the parsed anchors into the events[] the HTML renders."""
    ev = []
    us = turn["user_started"] if turn["user_started"] is not None else -2.0
    sents = turn["sentences"]
    ttfb = turn["tts_ttfb"]
    proc = turn["tts_proc"]
    bs = turn["bot_started"]
    bstop = turn["bot_stopped"] if turn["bot_stopped"] is not None else bs

    ev.append(dict(stage="capture", t=us, end=0, kind="span", label="User speaking",
                   why="Browser mic -> WebRTC -> Silero VAD listens for the end of the utterance.",
                   src="log user-turn-started"))
    ev.append(dict(stage="stt", t=us, end=0, kind="span", label="Deepgram nova-2 streaming (receives mic)",
                   why="STT's input is the live mic the whole time the user talks; partials refine into one final transcript.",
                   src="log STT stream"))
    ev.append(dict(stage="capture", t=0, kind="turn", label="User STOPPED speaking - t0",
                   why="VAD + turn-analyzer agree the turn ended. This instant starts the <3s TTFO stopwatch.",
                   src="log user-turn-stopped"))
    ev.append(dict(stage="stt", t=0, kind="emit", label="STT emits final transcript",
                   why=(f"\"{turn['question']}\" " if turn["question"] else "") + "pushed into the LLM context aggregator.",
                   src="log t0"))
    ev.append(dict(stage="llm", t=0, kind="recv", **{"from": "STT"}, label=f"{RING} LLM receives the transcript",
                   why="Generation starts at t0; the LLM was pre-warmed on connect, so no cold start.",
                   src="log 'Generating chat from context'"))
    if turn["llm_ttfb"]:
        lt, lv = turn["llm_ttfb"]
        ev.append(dict(stage="llm", t=0, end=lt, kind="span", label="OpenRouter generating", subtle=True,
                       why="Streams tokens; the whole answer is ready well before TTS finishes sentence 1.",
                       src="log LLM"))
        ev.append(dict(stage="llm", t=lt, kind="emit", label="LLM emits first token",
                       why=f"OpenRouter TTFB {lv:.3f}s - the transpacific cloud hop is the LLM's main cost.",
                       src=f"log LLM TTFB {lv:.3f}s"))

    # Per-sentence TTS synthesis bars + emit dots + recv markers (serial).
    for i, (st, text) in enumerate(sents):
        # this sentence's done time = next sentence's start, else last processing time
        done = sents[i + 1][0] if i + 1 < len(sents) else (proc[-1] if proc else st)
        fb = ttfb[i] if i < len(ttfb) else None
        ev.append(dict(stage="tts", t=st, kind="recv", **{"from": "LLM"},
                       label=f"{RING} TTS receives sentence {i+1}",
                       why=(f"\"{text}\" " if i == 0 else "") +
                           ("Starts as the previous sentence finished - CosyVoice synthesizes one sentence at a time (serial)."
                            if i > 0 else "First complete sentence flushed early so TTS can start before the full answer exists."),
                       src="log run_tts"))
        ev.append(dict(stage="tts", t=st, end=done, kind="span", label=f"CosyVoice synthesizing sentence {i+1}",
                       why=(f"First audio at TTFB {fb[1]:.3f}s; " if fb else "") +
                           (f"done at +{done:.2f}s. " ) +
                           ("This first chunk is the single biggest piece of TTFO." if i == 0
                            else "Synthesized while earlier audio is already playing, so it doesn't affect TTFO."),
                       src="log run_tts->TTFB"))
        if fb:
            ev.append(dict(stage="tts", t=fb[0], kind="emit", label=f"TTS emits sentence-{i+1} first chunk",
                           why=f"CosyVoice TTFB {fb[1]:.3f}s after receiving the sentence." +
                               (" This is what starts the bot speaking." if i == 0 else ""),
                           src=f"log TTS TTFB {fb[1]:.3f}s"))

    # avatar idle + voice receipt + render
    first_tts_fb = turn["tts_ttfb"][0][0] if turn["tts_ttfb"] else bs
    ev.append(dict(stage="avatar", t=0, end=bs, kind="span", label="Idle / neutral frames", subtle=True,
                   why="A calm neutral face between turns (real-time fps) so the picture is never frozen while TTS works.",
                   src="probe: video pre-speech"))
    ev.append(dict(stage="avatar", t=first_tts_fb, kind="recv", **{"from": "CosyVoice"},
                   label=f"{RING} MuseTalk receives the voice",
                   why="The avatar forwards the first PCM chunk to the :8002 render server the instant TTS emits it (real-time-paced).",
                   src="~ TTS first chunk"))
    ev.append(dict(stage="deliver", t=bs, kind="turn", big=True, label=f"Bot started speaking -> TTFO {turn['ttfo_s']}s",
                   why="The VOICE starts reaching the client here - audio is forwarded immediately WITHOUT waiting for a rendered frame. "
                       f"TTFO measures this audio start: {turn['ttfo_s']}s vs the 3s target. Lip-synced video is rendered best-effort (decoupled).",
                   src="log [TTFO] (audio-path event)"))
    ev.append(dict(stage="avatar", t=bs, end=bstop, kind="span", label="MuseTalk lip-sync render",
                   why="Mouth-region frames, live/audio-master sync: the voice is forwarded immediately so it can never freeze; lips track best-effort on the shared GPU.",
                   src="log render window"))
    ev.append(dict(stage="deliver", t=bstop, kind="turn", label="Bot stopped speaking - turn complete",
                   why="Full answer delivered. Mic un-mutes; the assistant aggregator records the turn so the next turn has full history.",
                   src="log bot-stopped"))
    return ev


def build_handoffs(turn):
    sents = turn["sentences"]
    first_tts_fb = turn["tts_ttfb"][0][0] if turn["tts_ttfb"] else turn["bot_started"]
    s1 = sents[0][0] if sents else 0.0
    return [
        dict(**{"from": "stt"}, to="llm", t=0.0, what="final transcript (text)",
             note="The question text enters the LLM at t0; generation starts immediately (pre-warmed)."),
        dict(**{"from": "llm"}, to="tts", t=s1, what="sentence 1 (text)",
             note="First complete sentence flushed early, so speech can begin before the full answer exists."),
        dict(**{"from": "tts"}, to="avatar", t=first_tts_fb, what="first voice chunk (16kHz PCM)", star=True,
             note="CosyVoice -> MuseTalk: the avatar forwards the chunk to the :8002 render server the instant TTS emits it."),
        dict(**{"from": "avatar"}, to="deliver", t=turn["bot_started"], what="voice starts (audio forwarded)",
             note="MuseTalk -> browser: the VOICE starts immediately - audio is NOT gated on a rendered frame. This is TTFO. "
                  "Lip frames are rendered best-effort and can lag the voice under GPU load."),
    ]


def build_metrics(turn, pm, offline_lip):
    def tag(cond_ok, cond_warn=False):
        return "ok" if cond_ok else ("warn" if cond_warn else "bad")
    M = []
    M.append(dict(k="TTFO", v=str(turn["ttfo_s"]), u="s", n="target 3s",
                  tag="ok" if turn["ttfo_pass"] else "bad"))
    if "startup_s" in pm:
        M.append(dict(k="Startup (connect -> 1st frame)", v=str(pm["startup_s"]), u="s", n="incl. idle warmup", tag=""))
        M.append(dict(k="Received video", v=str(pm["recv_fps"]), u="fps", n="server output rate", tag="ok"))
        M.append(dict(k="Frame interval", v=str(pm["frame_ms_mean"]), u="ms mean",
                      n=f"p95 {pm['frame_ms_p95']} - max {pm['frame_ms_max']}", tag=""))
        M.append(dict(k="Freeze (max gap)", v=str(pm["freeze_ms"]), u="ms",
                      n="OK <500ms" if pm["freeze_ms"] < 500 else "FAIL >500ms",
                      tag=tag(pm["freeze_ms"] < 500)))
    if "audio_gap_max_ms" in pm:
        M.append(dict(k="Audio arrival gap", v=str(pm["audio_gap_max_ms"]), u="ms max",
                      n=f"p95 {pm['audio_gap_p95_ms']}ms",
                      tag=tag(pm["audio_gap_max_ms"] < 50, pm["audio_gap_max_ms"] < 80)))
    # lip offset: prefer the offline (guaranteed-video) measurement
    lip = offline_lip if offline_lip else (
        dict(ms=pm["lip_offset_ms"], corr=pm.get("lip_offset_corr")) if "lip_offset_ms" in pm else None)
    if lip:
        sign = "lips lag" if lip["ms"] > 0 else "lips lead"
        src = "offline" if offline_lip else "webrtc"
        corr = lip.get("corr")
        if corr is not None and corr < 0.3:
            # xcorr found no real peak — do NOT present this as a trustworthy offset.
            M.append(dict(k="Lip offset", v=f"{lip['ms']:+d}", u="ms",
                          n=f"{sign}, corr {corr} - LOW-CONF / unreliable ({src})", tag=""))
        else:
            M.append(dict(k="Lip offset", v=f"{lip['ms']:+d}", u="ms",
                          n=f"{sign}, corr {corr} ({src})",
                          tag=tag(abs(lip["ms"]) < 80, abs(lip["ms"]) < 150)))
    else:
        M.append(dict(k="Lip offset", v="n/a", u="", n=pm.get("lip_offset_note", "unavailable"), tag=""))
    M.append(dict(k="Frames / audio pkts", v=str(pm.get("video_frames", 0)),
                  u=f"/ {pm.get('audio_packets', 0)}", n="over the capture", tag=""))
    return M


# --------------------------------------------------------- optional offline cap
async def offline_capture(mic_wav: str, fps: int):
    """Drive the MuseTalk server directly -> mp4 WITH a video track -> clean lip offset.
    Runs AFTER the probe has disconnected (the server is single-client)."""
    import wave
    import cv2  # noqa: F401  (only needed if we ever rewrite frames; ffmpeg path below)
    import websockets

    out_mp4 = str(ROOT / "output" / "measure_offline.mp4")
    silent = str(ROOT / "output" / "measure_offline_silent.mp4")
    with wave.open(mic_wav, "rb") as w:
        sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    if sr != 16000:
        a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)),
                      np.arange(len(a)), a).astype(np.int16)
    frames = []
    try:
        async with websockets.connect("ws://localhost:8002/stream", max_size=None) as ws:
            async def reader():
                try:
                    while True:
                        m = await ws.recv()
                        if isinstance(m, (bytes, bytearray)):
                            frames.append(bytes(m))
                except Exception:
                    pass
            rt = asyncio.create_task(reader())
            await ws.send(json.dumps({"type": "config", "fps": fps}))
            await ws.send(json.dumps({"type": "speech_start"}))
            ch = int(16000 * 0.02)
            for i in range(0, len(a), ch):
                await ws.send(a[i:i + ch].tobytes())
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "speech_end"}))
            await asyncio.sleep(2.0)
            rt.cancel()
    except Exception as e:  # noqa: BLE001
        print(f"  offline-capture skipped ({e!r})")
        return None
    if len(frames) < 10:
        print("  offline-capture: too few frames"); return None
    import cv2
    import subprocess
    side = int(round((len(frames[0]) / 3) ** 0.5))
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (side, side))
    for fb in frames:
        arr = np.frombuffer(fb, np.uint8).reshape(side, side, 3)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    vw.release()
    ffmpeg = r"E:\miniconda3\envs\tts\Library\bin\ffmpeg.exe"
    try:
        subprocess.run([ffmpeg, "-y", "-i", silent, "-i", mic_wav, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_mp4],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        print(f"  offline-capture mux failed ({e!r})"); return None
    off, corr, err = lip_offset_from_mp4(out_mp4, fps)
    if err:
        print(f"  offline lip offset unavailable ({err})"); return None
    print(f"  offline capture: {len(frames)} frames -> {out_mp4}")
    return dict(ms=round(off * 1000), corr=round(corr, 2))


# ----------------------------------------------------------------------- output
def write_outputs(report):
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    js = "// AUTO-GENERATED by scripts/measure.py — do not edit by hand.\n"
    js += "window.MEASURE = " + json.dumps(report, ensure_ascii=False) + ";\n"
    JS_OUT.write_text(js, encoding="utf-8")


def print_summary(report):
    t = report["meta"]
    print("\n==================== MEASURE REPORT ====================")
    print(f"turn: \"{t.get('question')}\"  @ {t['when']}")
    print(f"TTFO: {t['ttfo']}s (target {t['ttfo_target']}s) {'PASS' if t['ttfo_pass'] else 'FAIL'}")
    print("handoffs (node receives input @):")
    for h in report["handoffs"]:
        print(f"  {h['from']:>8} -> {h['to']:<8} +{h['t']:.2f}s   {h['what']}")
    print("metrics:")
    for m in report["metrics"]:
        print(f"  {m['k']:<28} {m['v']} {m['u']}  ({m['n']})")
    print("latency waterfall (t0 = user stopped -> user hears):")
    for r in report.get("waterfall", []):
        d = f"{r['delta']:+.2f}s" if r["delta"] is not None else "   ?  "
        c = f"{r['cum']:.2f}s" if r["cum"] is not None else "  ?  "
        src = f"[{r['source']}]" if r["source"] else ""
        print(f"  {r['stage']:<36} {d:>8}   cum {c:>7}  {src}")
    print(f"wrote {JSON_OUT}")
    print(f"wrote {JS_OUT}  -> open docs/workflow-timeline.html (auto-uses it)")
    print("=======================================================\n")


async def main(args):
    lines_cache = None
    if args.from_browser:
        print("[1/3] --from-browser: parsing the last real-browser turn (no headless probe)...")
        vwall, awall, pm = [], [], {}
    else:
        print("[1/3] driving a real turn through the live pipeline (WebRTC)...")
        vwall, awall, connect_t = await run_probe(args.mic, args.lead, args.tail, args.duration)
        pm = probe_metrics(vwall, awall, connect_t, args.fps)

    print("[2/3] parsing the pipeline.log delta for this turn...")
    turn = None
    if args.from_browser:
        # Browser turns log no [TTFO, so anchor on the real [client-playout] beacon instead of
        # letting parse_turn() lock onto a stale headless [TTFO turn.
        turn = parse_browser_turn()
        if turn is None:
            print("  [warn] no [client-playout] beacon found — falling back to the last [TTFO turn. "
                  "Hard-reload /client/ (so the probe injects), do ONE turn, then re-run.")
    if turn is None:
        turn = parse_turn()
    t0_epoch = turn["t0"].timestamp()

    # Last mile source 1 (headless): first ANSWER audio arriving at the client, on the log clock.
    # Guard: parse_turn() returns the LAST [TTFO] in the log, which can be an OLD turn if the probe
    # never drove a fresh one. Correlating this run's live audio arrivals against a stale t0 yields a
    # plausible-but-wrong number, so only trust the arrival when t0 falls inside this capture window.
    client_arrival = None
    if awall:
        onset = answer_onset_epoch(awall, t0_epoch)
        turn_age = time.time() - t0_epoch
        fresh_window = args.duration + args.tail + 15.0
        if onset is None:
            pass
        elif turn_age > fresh_window:
            print(f"  [warn] latest log turn is {turn_age:.0f}s old (> {fresh_window:.0f}s window) "
                  "-- no fresh turn captured; client_arrival left blank.")
        else:
            client_arrival = round(onset - t0_epoch, 3)

    # Last mile source 2 (real browser): the [client-playout] onset beacon, if present.
    playout = None
    if args.from_browser:
        if turn.get("playout_epoch") is not None:
            # Use the exact beacon parse_browser_turn anchored on (avoids re-matching a stale/bogus
            # beacon logged after t0). Its browser onset epoch is on the same-box clock as t0.
            playout = round(turn["playout_epoch"] - t0_epoch, 3)
        else:
            lines_cache = _parse_lines()
            playout = parse_playout_beacon(lines_cache, turn["t0"])

    anchors = dict(
        llm_recv=0.0,
        llm_ttfb=turn["llm_ttfb"][0] if turn["llm_ttfb"] else None,
        tts_recv=turn["sentences"][0][0] if turn["sentences"] else None,
        tts_ttfb=turn["tts_ttfb"][0][0] if turn["tts_ttfb"] else None,
        bot_started=turn["bot_started"],
        client_arrival=client_arrival,
        playout=playout,
    )
    # Fill the playout row: measured browser beacon, else estimate = arrival + jitter buffer.
    if anchors["playout"] is not None:
        # When there's no headless client_arrival (browser-only mode), the transport row is
        # 'unknown' so this beacon Delta absorbs server->browser transport+network too -- say so.
        playout_source = "browser" if client_arrival is not None else "browser+net"
    elif client_arrival is not None:
        jb = float(os.getenv("CLIENT_JITTER_BUFFER_MS", "400") or 400) / 1000.0
        anchors["playout"] = round(client_arrival + jb, 3)
        playout_source = "est"
    else:
        playout_source = "est"

    offline_lip = None
    if args.offline_capture and not args.from_browser:
        ow = args.offline_wav if Path(args.offline_wav).exists() else args.mic
        print(f"[3/3] offline avatar capture for a clean lip offset (wav={ow})...")
        offline_lip = await offline_capture(ow, args.fps)
    else:
        print("[3/3] offline capture skipped.")

    report = {
        "meta": {
            "when": turn["t0"].strftime("%Y-%m-%d %H:%M"),
            "question": turn["question"],
            "machine": args.machine,
            "stack": args.stack,
            "ttfo": turn["ttfo_s"], "ttfo_target": 3.0, "ttfo_pass": turn["ttfo_pass"],
        },
        "events": build_events(turn),
        "handoffs": build_handoffs(turn),
        "metrics": build_metrics(turn, pm, offline_lip),
        "waterfall": build_waterfall(anchors, playout_source),
        "raw": {"probe": pm, "ttfo_s": turn["ttfo_s"], "anchors": anchors,
                "sentences": turn["sentences"], "tts_ttfb": turn["tts_ttfb"]},
    }
    write_outputs(report)
    print_summary(report)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 crashes on glyphs
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Unified VisualLLm turn measurement.")
    ap.add_argument("--mic", default="output/q_ai.wav")
    ap.add_argument("--lead", type=float, default=8.0)
    ap.add_argument("--tail", type=float, default=28.0)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--offline-capture", action="store_true",
                    help="also drive the MuseTalk server directly for a guaranteed lip offset")
    ap.add_argument("--from-browser", action="store_true",
                    help="parse-only: use a real browser's [client-playout] beacon for the last "
                         "mile instead of driving the headless probe (open /client, do one turn, "
                         "with CLIENT_PLAYOUT_PROBE=1)")
    ap.add_argument("--offline-wav", default="output/reply_concise.wav",
                    help="wav for the offline avatar capture; a longer bot-reply clip gives a more reliable lip offset")
    ap.add_argument("--machine", default="this box (RTX 5060 Ti, Blackwell)")
    ap.add_argument("--stack", default="Deepgram STT - OpenRouter LLM - CosyVoice2 (vLLM/WSL) TTS - MuseTalk avatar")
    asyncio.run(main(ap.parse_args()))
