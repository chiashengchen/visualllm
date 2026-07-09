"""MuseTalk local lip-sync avatar as a Pipecat FrameProcessor.

Sits between TTS and transport.output(). It streams the TTS audio to a local MuseTalk
server over a websocket, receives lip-synced RGB frames back, and pushes both video
(OutputImageRawFrame) and audio (TTSAudioRawFrame) downstream IN SYNC.

A/V sync is the hard part. The server renders with some
latency, so we cannot just forward the audio immediately and let the video free-run (that is
the desync). Instead the server emits markers -- `video_start` / `video_clock{frames:N}` /
`video_end` -- and this client buffers the voice and releases each audio chunk paired with its
matching rendered frame, tagging the frame `OutputImageRawFrame.sync_with_audio=True` so the
transport (non-live) pins each frame to its audio position. Two strategies (MUSETALK_SYNC_MODE):

  steady    : release incrementally as `video_clock` advances. Because MuseTalk renders
              steadily at ~real-time (no diffusion warmup), the clock advances smoothly so the
              voice plays smoothly -- low latency, no stutter.
  prerender : buffer the whole short reply, release it all aligned on `video_end` -- near-perfect
              sync at the cost of ~one render's worth of extra start delay.

Set MUSETALK_SYNC_WITH_AUDIO=0 to fall back to the old free-running behaviour.
Talks to local_services/musetalk_server/ (FastAPI + websocket). Requires `pip install websockets`.
"""
from __future__ import annotations

import asyncio
import json
import math
import os

import numpy as np
import websockets
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

MUSETALK_SR = 16000  # Whisper (server-side) expects 16 kHz mono


def _to_16k_mono_pcm(audio: bytes, in_rate: int, channels: int) -> bytes:
    """Resample int16 PCM to 16 kHz mono for the MuseTalk server."""
    if len(audio) & 1:
        audio = audio[:-1]  # pipecat's resampler can hand us an odd-length buffer; int16
        #                     needs an even byte count or np.frombuffer raises. Drop the
        #                     stray byte (sub-sample, inaudible) instead of dropping the chunk.
    a = np.frombuffer(audio, dtype=np.int16)
    if a.size == 0:
        return b""
    if channels and channels > 1:
        a = a.reshape(-1, channels).mean(axis=1)
    if in_rate and in_rate != MUSETALK_SR:
        n_out = int(round(a.shape[0] * MUSETALK_SR / in_rate))
        if n_out <= 0:
            return b""
        src = np.arange(a.shape[0], dtype=np.float64)
        dst = np.linspace(0, a.shape[0] - 1, num=n_out)
        a = np.interp(dst, src, a)
    return a.astype(np.int16).tobytes()


class MuseTalkVideoService(FrameProcessor):
    def __init__(
        self,
        *,
        base_url: str,
        fps: int = 20,
        image_size: tuple[int, int] = (512, 512),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws_url = base_url.replace("http", "ws", 1).rstrip("/") + "/stream"
        self._fps = float(fps)
        self._size = image_size
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._closing = False

        # --- sync config ---
        # MODE:
        #   live (default) = AUDIO-MASTER. The voice plays at real-time; lip-sync is best-effort
        #     and bounded -- we stop feeding the server when it falls > MAX_LAG behind, so on a
        #     slow/contended GPU the lips skip stale content to stay current instead of dragging
        #     the WHOLE voice slow/late. This is the only sane behaviour when the render can't
        #     sustain real-time (MuseTalk shares the GPU with CosyVoice).
        #   steady / prerender = VIDEO-MASTER (sync_with_audio pinning) -- tight sync ONLY when the
        #     render keeps up; on a slow GPU they make the voice lag. Kept for fast-GPU setups.
        self._mode = (os.getenv("MUSETALK_SYNC_MODE", "steady") or "steady").lower()
        self._sync = self._mode in ("steady", "prerender") and (
            os.getenv("MUSETALK_SYNC_WITH_AUDIO", "1") or "1").lower() in ("1", "true", "yes", "on")
        self._fallback_s = float(os.getenv("MUSETALK_SYNC_FALLBACK_S", "10.0"))
        self._last_hold_log = 0.0

        # Real-time-paced feed to the server (live mode). CosyVoice produces the whole reply
        # FASTER than real-time (RTF<1); if we forwarded all that audio to the renderer as fast
        # as it arrives, the server renders a big backlog that plays out at fps -> the video
        # trails the voice by seconds ("audio done, avatar still going"). So we release audio to
        # the server paced to real-time, keeping the render in lockstep with playback.
        self._feed_q: asyncio.Queue = asyncio.Queue()
        self._feed_task: asyncio.Task | None = None
        # Startup-latency fix: burst the first MUSETALK_FEED_BURST_S of a turn's audio to the
        # server WITHOUT real-time pacing, so it can render the opening frames immediately. The
        # lips otherwise start ~2s late because a fully real-time-paced feed STARVES the renderer
        # at turn start (it can't fill its lead-prime + first segment until audio trickles in).
        # After the burst we resume real-time pacing so no backlog builds (the original guarantee).
        self._burst_s = float(os.getenv("MUSETALK_FEED_BURST_S", "1.0"))
        self._burst_remaining = 0.0

        # --- per-turn sync state ---
        self._lock = asyncio.Lock()
        self._abuf: list[tuple[float, Frame, FrameDirection]] = []  # (cum_end_s, frame, dir)
        self._aidx = 0                # audio release cursor into _abuf
        self._audio_clock_s = 0.0     # seconds of audio buffered this turn
        self._vbuf: list[bytes] = []  # rendered frames this turn (index == real frame #)
        self._released_idx = 0        # video release cursor into _vbuf
        self._video_active = False    # between video_start and video_end
        self._unsynced = False        # fallback engaged this turn
        self._fallback_task: asyncio.Task | None = None

        # --- per-turn A/V timing instrumentation (logs audio-vs-avatar offset + lip drift) ---
        self._t_audio_first: float | None = None   # loop.time() of first voice chunk this turn
        self._t_vid_first: float | None = None      # loop.time() of first rendered frame this turn
        self._t_vid_last: float | None = None
        self._vframes = 0                           # real lip-synced frames this turn
        self._aud_dur = 0.0                         # seconds of voice this turn
        self._last_offset_log = 0.0                 # throttle for the continuous offset trace
        self._odd_carry = b""                       # anti-screech: dangling odd byte carried between
        #   downstream audio frames so the PCM stays whole-sample (see _align_even)

        # --- smooth end-of-turn close (steady) ---
        # MuseTalk can't ease the mouth shut itself (silence renders a PARTED mouth, not closed
        # lips -- measured), so at end of turn we cross-dissolve the last spoken frame -> the rest
        # pose over K frames. To survive steady's NON-LIVE transport (which drops audio-less
        # trailing frames) each close frame is paired with one frame of trailing SILENCE and pushed
        # through the SAME _emit_pair path as every speech frame. Gated by MUSETALK_CLOSE_FADE_FRAMES
        # (0 = off, the old clean snap). Use with MUSETALK_END_TAIL_FRAMES=0 so the last buffered
        # frame is the last SPOKEN frame, not a neutral tail copy.
        self._close_fade = int(os.getenv("MUSETALK_CLOSE_FADE_FRAMES", "0") or "0")
        self._rest_frame: bytes | None = None   # cached between-turn rest pose (crossfade target)
        self._tts_sr = MUSETALK_SR              # last turn's TTS sample rate (for the silence pad)
        self._tts_ch = 1
        self._suppress_until = 0.0              # drop server idle frames until here so they can't
        #   preempt the crossfade playout (the burst-flush collapse P12 hit)

        # --- freeze watchdog (capture the REAL freeze the ~1s hold/offset sampling misses) ---
        # A freeze = video frames stop reaching the transport for a beat. Track the wall-gap
        # between EMITTED OutputImageRawFrames (every release path funnels through push_frame) and
        # warn the instant it exceeds MUSETALK_STALL_LOG_S, classified render-starved (the server
        # also stopped feeding us -> high arrival gap) vs delivery-side (frames buffered but not
        # going out). A TOTAL stall emits nothing, so a poller (_watch_loop) raises it rather than
        # the next emit. Pairs with the browser-side monitor in main.py for the transport/browser
        # leg the server can't see. Default 0 (OFF) -- diagnostic scaffolding; set a value like
        # 0.4 to re-arm it when hunting a freeze.
        self._stall_s = float(os.getenv("MUSETALK_STALL_LOG_S", "0") or "0")
        self._last_emit_t: float | None = None   # loop.time() of the last video frame pushed out
        self._stall_open = False                 # a freeze is currently being reported
        self._watch_task: asyncio.Task | None = None

    # --- connection lifecycle ---------------------------------------------
    async def _connect(self):
        if self._recv_task is not None:
            return
        self._closing = False
        try:
            await self._open_ws()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MuseTalk initial connect failed ({e!r}); loop will retry.")
        self._recv_task = asyncio.create_task(self._receive_loop())
        if self._feed_task is None:
            self._feed_task = asyncio.create_task(self._feed_loop())
        if self._watch_task is None and self._stall_s > 0:
            self._watch_task = asyncio.create_task(self._watch_loop())

    async def _open_ws(self):
        logger.info(f"Connecting to MuseTalk server at {self._ws_url} "
                    f"(sync={'on:'+self._mode if self._sync else 'off'})")
        self._ws = await websockets.connect(
            self._ws_url, max_size=None, ping_interval=None, close_timeout=1
        )
        await self._ws.send(json.dumps({"type": "config", "fps": self._fps}))

    async def _feed_loop(self):
        """Send queued items to the server, pacing AUDIO to real-time so the renderer never
        builds a backlog (the cause of the voice-finishes-but-video-keeps-going lag). Markers
        (start/end/reset) are forwarded immediately, in order with the audio."""
        while not self._closing:
            try:
                kind, payload = await self._feed_q.get()
            except asyncio.CancelledError:
                break
            try:
                if self._ws is None:
                    continue
                if kind == "audio":
                    pcm, dur = payload
                    await self._ws.send(pcm)
                    if self._burst_remaining > 0:
                        self._burst_remaining -= dur   # BURST: skip the pace so the renderer can
                        #   start the opening frames immediately (kills the ~2s startup starve)
                    else:
                        await asyncio.sleep(dur)        # then pace to real-time (no backlog)
                else:
                    if kind == "speech_start":
                        self._burst_remaining = self._burst_s   # reset the burst budget per turn
                    await self._ws.send(json.dumps({"type": kind}))
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001
                pass

    async def _disconnect(self):
        self._closing = True
        self._cancel_fallback()
        for task_attr in ("_recv_task", "_feed_task", "_watch_task"):
            task = getattr(self, task_attr)
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                setattr(self, task_attr, None)
        if self._ws:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None

    async def _receive_loop(self):
        while not self._closing:
            try:
                if self._ws is None:
                    await self._open_ws()
                assert self._ws is not None
                async for message in self._ws:
                    if isinstance(message, bytes):
                        await self._on_frame(message)
                    else:
                        await self._on_marker(json.loads(message))
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                if not self._closing:
                    logger.warning(f"MuseTalk ws dropped ({e!r}); reconnecting...")
            finally:
                ws, self._ws = self._ws, None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:  # noqa: BLE001
                        pass
            if not self._closing:
                await asyncio.sleep(0.5)

    # --- sync core --------------------------------------------------------
    async def _on_frame(self, img: bytes):
        """A rendered RGB frame from the server."""
        if self._video_active:   # count frames + trace the live lips-vs-voice offset
            now = asyncio.get_running_loop().time()
            if self._t_vid_first is None:
                self._t_vid_first = now
            self._t_vid_last = now
            self._vframes += 1
            # Continuous (delivery-side) offset: real-time voice elapsed vs lip-video delivered.
            # + = lips behind the voice, - = lips ahead. Shows the swing within a turn (the burst
            # can push the lips AHEAD after they start behind). NOTE: this is delivery to the
            # transport, not browser playout (the jitter buffer smooths some of it).
            if self._t_audio_first is not None and now - self._last_offset_log >= 1.0:
                self._last_offset_log = now
                voice_s = now - self._t_audio_first
                video_s = self._vframes / self._fps
                off = voice_s - video_s
                tag = f"{off:+0.2f}s behind" if off > 0.05 else (
                    f"{-off:0.2f}s AHEAD" if off < -0.05 else "in step")
                logger.info(f"[avatar offset] {voice_s:0.1f}s in: lips {tag}")
        if not self._sync:
            await self.push_frame(
                OutputImageRawFrame(image=img, size=self._size, format="RGB"),
                FrameDirection.DOWNSTREAM,
            )
            return
        if self._video_active and not self._unsynced:
            async with self._lock:
                self._vbuf.append(img)
            if self._mode == "steady":
                await self._advance()   # release paced to frames AS they arrive (continuous)
        else:
            # idle frame (between turns) or fallback: animate immediately, untagged.
            if not self._video_active:
                # Cache the rest pose (crossfade target) and, while a close crossfade is playing
                # out, DROP these server idle frames so they can't preempt it (the burst-flush
                # collapse P12 hit -- the transport's current image would jump to neutral).
                self._rest_frame = img
                if asyncio.get_running_loop().time() < self._suppress_until:
                    return
            await self.push_frame(
                OutputImageRawFrame(image=img, size=self._size, format="RGB"),
                FrameDirection.DOWNSTREAM,
            )

    async def _on_marker(self, evt: dict):
        kind = evt.get("type")
        if kind == "video_start":
            # New turn segment. TTSStartedFrame already reset the turn (audio + buffers) BEFORE
            # any audio was buffered, so here we ONLY mark active. We deliberately do NOT clear
            # _vbuf / _released_idx -- they stay continuous across the whole turn so a stray
            # mid-reply re-segment can't desync the frame<->audio mapping.
            self._cancel_fallback()
            self._video_active = True
        elif kind == "video_clock":
            if not self._unsynced and self._mode == "steady":
                await self._advance()   # heartbeat (real pacing is on frame receipt)
        elif kind == "video_end":
            close_start = self._vbuf[-1] if self._vbuf else None   # last spoken frame (crossfade src)
            await self._advance()       # flush whatever is buffered (prerender: the whole reply)
            await self._drain_audio()   # release the turn's trailing voice
            self._log_turn_timing()
            self._video_active = False
            if self._close_fade > 0 and close_start is not None and self._rest_frame is not None:
                # Ease the mouth shut: FREE-RUN the crossfade (untagged, like the idle loop) so it
                # is NOT gated by the audio-cap in _advance -- that cap strands trailing frames when
                # the render ran behind (video > audio). Suppress server idle frames during the
                # playout so they can't preempt it; the fade lands on the rest pose, so the neutral
                # the server then holds is seamless. ("Live during the close" within steady.)
                self._suppress_until = (asyncio.get_running_loop().time()
                                        + self._close_fade / self._fps + 0.3)
                asyncio.create_task(self._play_close_fade(close_start, self._rest_frame))

    async def _advance(self):
        """Release received frames, each paired (in order) with the audio due by its time and
        tagged sync_with_audio so the transport pins it. Never release past the voice we have
        buffered (the audio_cap below), so the video can't run ahead of the voice."""
        async with self._lock:
            # Never release video past the voice we actually have buffered: a frame at index
            # i needs the audio up to i/fps, so cap at the buffered-audio position. This stops
            # the video running AHEAD of the voice (e.g. if frames briefly outpace audio).
            target = len(self._vbuf)
            # ceil (not floor): a turn's audio is rarely a whole number of frames
            # (13.6s*12 = 163.2), so int()/floor stranded the final sub-frame of audio to the
            # delayed video_end drain -> it played ~1s late as a blip (PROBLEMS-AND-FIXES P10).
            # ceil makes the trailing frame reachable so the last sub-frame releases in step.
            audio_cap = math.ceil(self._audio_clock_s * self._fps) + 1
            target = min(target, audio_cap)
            while self._released_idx < target:
                await self._emit_pair(self._released_idx)
                self._released_idx += 1
            self._log_hold()

    async def _emit_pair(self, i: int):
        """Audio due by frame i's time (in order), then frame i tagged sync_with_audio. Caller
        holds the lock. Frame is skipped if not buffered yet (a caught-up cursor ran ahead)."""
        ft = i / self._fps
        while self._aidx < len(self._abuf) and self._abuf[self._aidx][0] <= ft:
            _e, af, ad = self._abuf[self._aidx]
            self._aidx += 1
            await self.push_frame(af, ad)
        if i < len(self._vbuf):
            fr = OutputImageRawFrame(image=self._vbuf[i], size=self._size, format="RGB")
            fr.sync_with_audio = True
            await self.push_frame(fr, FrameDirection.DOWNSTREAM)

    def _log_hold(self):
        now = asyncio.get_running_loop().time()
        if now - self._last_hold_log >= 1.0:
            self._last_hold_log = now
            hold = self._audio_clock_s - self._released_idx / self._fps
            logger.info(
                f"[musetalk sync] hold={hold:0.2f}s (audio {self._audio_clock_s:0.1f}s, "
                f"video {self._released_idx/self._fps:0.1f}s) "
                f"abuf={len(self._abuf)-self._aidx} vbuf={len(self._vbuf)-self._released_idx}"
            )

    async def _drain_audio(self):
        """Release any audio left after the last rendered frame (tail of the turn)."""
        async with self._lock:
            while self._aidx < len(self._abuf):
                _e, af, ad = self._abuf[self._aidx]
                self._aidx += 1
                await self.push_frame(af, ad)

    async def _play_close_fade(self, last_bytes: bytes, rest_bytes: bytes):
        """Free-run a pixel cross-dissolve (last spoken frame -> rest pose) at fps so the mouth
        eases shut at end of turn. Untagged frames (like the idle loop) so the non-live transport
        draws them on its own clock -- NOT audio-paired, so the _advance audio-cap can't strand
        them when the render fell behind (video > audio). We blend PIXELS because MuseTalk renders
        silence as a PARTED mouth, not closed (measured), so feeding it can't close the mouth. The
        final blended frame == the rest pose, so the neutral the server holds afterwards is seamless.
        Best-effort: any failure just leaves the prior clean snap."""
        try:
            last = np.frombuffer(last_bytes, dtype=np.uint8).astype(np.float32)
            rest = np.frombuffer(rest_bytes, dtype=np.uint8).astype(np.float32)
            if last.shape != rest.shape:
                return
            interval = 1.0 / self._fps
            for j in range(1, self._close_fade + 1):
                a = j / self._close_fade                    # linear: lands exactly on the rest pose
                blended = ((1.0 - a) * last + a * rest).astype(np.uint8).tobytes()
                await self.push_frame(
                    OutputImageRawFrame(image=blended, size=self._size, format="RGB"),
                    FrameDirection.DOWNSTREAM)
                await asyncio.sleep(interval)
        except Exception:  # noqa: BLE001 -- close polish only; never disrupt the next turn
            pass

    def _reset_turn(self):
        self._abuf = []
        self._aidx = 0
        self._audio_clock_s = 0.0
        self._vbuf = []
        self._released_idx = 0
        self._video_active = False
        self._unsynced = False
        self._t_audio_first = None
        self._t_vid_first = None
        self._t_vid_last = None
        self._vframes = 0
        self._aud_dur = 0.0
        self._last_offset_log = 0.0
        self._odd_carry = b""

    def _log_turn_timing(self):
        """Log this turn's audio-vs-avatar timing: how long after the voice the lips started,
        and how far the video fell behind by the end (the live lip drift). Best-effort."""
        if self._t_vid_first is None or self._t_audio_first is None:
            return
        startup = self._t_vid_first - self._t_audio_first
        vid_dur = self._vframes / self._fps if self._fps else 0.0
        span = (self._t_vid_last - self._t_vid_first) if self._t_vid_last else 0.0
        eff_fps = self._vframes / span if span > 0 else 0.0
        drift = self._aud_dur - vid_dur
        # The perceived lip lag is dominated by how late the lips START after the voice
        # (startup), plus any accumulating drift over the turn. Both must be small to be in step.
        verdict = "LIPS BEHIND" if (startup > 0.15 or drift > 0.15) else "in step"
        logger.info(
            f"[avatar timing] lips start +{startup:0.2f}s after voice | "
            f"audio {self._aud_dur:0.2f}s video {vid_dur:0.2f}s "
            f"({self._vframes} frames, {eff_fps:0.1f} fps) | "
            f"end drift +{drift:0.2f}s -> {verdict}"
        )

    # --- fallback (marker-less server / lost markers) ---------------------
    def _arm_fallback(self):
        if self._fallback_task or not self._sync:
            return
        self._fallback_task = asyncio.create_task(self._fallback_watch())

    def _cancel_fallback(self):
        if self._fallback_task:
            self._fallback_task.cancel()
            self._fallback_task = None

    async def _fallback_watch(self):
        try:
            await asyncio.sleep(self._fallback_s)
        except asyncio.CancelledError:
            return
        if not self._video_active and self._abuf and self._aidx < len(self._abuf):
            logger.warning(f"MuseTalk sync: no video markers within {self._fallback_s}s; "
                           "forwarding voice unsynced for this turn.")
            self._unsynced = True
            await self._drain_audio()

    async def _reset_session(self):
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "reset"}))
            except Exception:  # noqa: BLE001
                pass

    def _align_even(self, frame: TTSAudioRawFrame) -> None:
        """SAMPLE-ALIGNMENT GUARD (the real steady-screech root-cause fix).

        The screech was an ODD-byte misalignment: pipecat's output transport clears its internal
        `_audio_buffer` (bytearray()) on `_bot_stopped_speaking` -- fired by a >3s render-stall gap
        OR by the per-turn TTSStoppedFrame. If the bytes accumulated there are an ODD count, the
        discard shifts every following int16 sample by half a sample -> loud broadband noise to the
        end of the turn. The odd count comes from CosyVoice's LAST chunk per utterance, which
        `iter_chunked` can hand back at an odd length.

        Fix at the source of truth: make EVERY audio frame we push downstream an even (whole-sample)
        byte count, carrying any dangling odd byte to the next frame so the PCM stream stays exactly
        contiguous. Then the transport's running total is always even, so any buffer-clear can only
        ever drop an even (whole-sample) gap -- at worst an inaudible click, NEVER the screech.
        Carry is per-connection and reset each turn (a dangling final byte is inaudible)."""
        data = self._odd_carry + frame.audio
        if len(data) & 1:
            self._odd_carry = data[-1:]
            data = data[:-1]
        else:
            self._odd_carry = b""
        frame.audio = data

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSAudioRawFrame):
            self._align_even(frame)   # keep the downstream PCM whole-sample (anti-screech guard)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, OutputImageRawFrame):
            self._note_emit()         # freeze watchdog: a video frame is leaving downstream
        await super().push_frame(frame, direction)

    def _note_emit(self):
        """Mark a video frame's departure; close out any freeze the watchdog opened (the gap
        since the last emit IS the freeze duration, so log it before overwriting the timestamp)."""
        if self._stall_s <= 0:
            return
        now = asyncio.get_running_loop().time()
        if self._stall_open and self._last_emit_t is not None:
            logger.warning(f"[avatar FREEZE] recovered after {(now - self._last_emit_t) * 1000:.0f}ms")
            self._stall_open = False
        self._last_emit_t = now

    async def _watch_loop(self):
        """Poll for a video-out stall the ~1s hold/offset logs can't see: if no frame has gone
        downstream for MUSETALK_STALL_LOG_S while a turn is live (or its audio still draining),
        log it ONCE with the state that localizes it (render-starved vs delivery-side)."""
        while not self._closing:
            await asyncio.sleep(0.2)
            if self._last_emit_t is None or self._stall_open:
                continue
            if not (self._video_active or self._aidx < len(self._abuf)):
                continue   # between turns, holding the rest pose is not a freeze
            now = asyncio.get_running_loop().time()
            gap = now - self._last_emit_t
            if gap < self._stall_s:
                continue
            arr = (now - self._t_vid_last) if self._t_vid_last is not None else -1.0
            cause = ("render-starved (server not sending frames)" if arr >= self._stall_s
                     else "delivery-side (frames buffered, not going downstream)")
            logger.warning(
                f"[avatar FREEZE] no video out for {gap * 1000:.0f}ms -> {cause}; "
                f"server-arrival-gap={arr * 1000:.0f}ms "
                f"vbuf={len(self._vbuf) - self._released_idx} abuf={len(self._abuf) - self._aidx} "
                f"active={self._video_active}"
            )
            self._stall_open = True

    # --- frame processing --------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._connect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._disconnect()
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            # Barge-in: drop any audio still queued for the server so the interrupted turn
            # can't keep driving the lips, then reset.
            while not self._feed_q.empty():
                try:
                    self._feed_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await self._reset_session()
            self._cancel_fallback()
            self._reset_turn()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            sr = getattr(frame, "sample_rate", MUSETALK_SR) or MUSETALK_SR
            ch = getattr(frame, "num_channels", 1) or 1
            self._tts_sr, self._tts_ch = sr, ch   # for the close crossfade's silence pad
            dur = (len(frame.audio) // (2 * ch)) / sr
            if self._t_audio_first is None:   # first voice chunk of the turn = audio start
                self._t_audio_first = asyncio.get_running_loop().time()
            self._aud_dur += dur
            # ALWAYS feed the server REAL-TIME-PACED (via _feed_q) so the renderer can't build a
            # backlog from CosyVoice's faster-than-real-time output (the "voice finishes but the
            # avatar keeps going" lag). Pacing keeps the server's queue ~empty either mode.
            pcm = _to_16k_mono_pcm(frame.audio, sr, ch)
            if pcm:
                self._feed_q.put_nowait(("audio", (pcm, dur)))
            if not self._sync:
                # AUDIO-MASTER (live): forward the voice NOW (plays at real-time, lips best-effort).
                await self.push_frame(frame, direction)
            elif self._unsynced:
                await self.push_frame(frame, direction)   # fallback: marker-less server
            else:
                # READINESS-GATED (steady): hold the voice and release it locked to the real
                # rendered frames -- the voice waits until the avatar is ready, then they play
                # together. No drift, no end cut. (See _advance / video_clock handling.)
                # The mid-speech "screech" that steady used to hit is NOT a held-frame problem --
                # it was pipecat discarding a partial (odd) audio buffer; fixed by _align_even (every
                # downstream frame kept whole-sample) + the BOT_VAD_STOP_FALLBACK_SECS raise in
                # main.py. So we just buffer the frame as-is here.
                self._audio_clock_s += dur
                async with self._lock:
                    self._abuf.append((self._audio_clock_s, frame, direction))
                self._arm_fallback()

        elif isinstance(frame, TTSStartedFrame):
            self._cancel_fallback()
            self._reset_turn()
            # speech_start/end go through _feed_q so they order correctly with the real-time-
            # paced audio (start before the turn's audio, end after it fully drains).
            self._feed_q.put_nowait(("speech_start", None))
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            self._feed_q.put_nowait(("speech_end", None))
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
