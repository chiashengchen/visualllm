"""Pipeline assembly + runner.

Order of frames per turn:
  mic -> transport.input() -> [Silero VAD on input] -> STT -> user context
       -> LLM (streamed, sentence-aggregated) -> TTS -> Avatar(lip-sync)
       -> TtfoMeter -> transport.output() -> browser (video+audio)

Run locally:
  python -m pipeline.main
then open the printed http://localhost URL in a browser.

This targets a recent Pipecat (uses the development runner + SmallWebRTC). If an
import path errors, check it against your installed version — the fragile bits
are isolated to the stage factories and the imports at the top here.
"""
from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipeline.config import config
from pipeline.metrics import TtfoMeter
from pipeline.stages import build_avatar, build_llm, build_stt, build_tts, build_vad_params
from local_services.avatar_memory import MemoryStore


async def run_bot(transport: BaseTransport, conn=None) -> None:
    # Pipecat's runner removes loguru sinks when main() starts, dropping the file
    # sink added in __main__ -> logs/pipeline.log would miss all runtime logs. This
    # runs after the runner has configured logging, so it re-asserts the file sink.
    from log_setup import ensure_file_sink

    ensure_file_sink("pipeline")

    _llm_label = "WeatherChain" if config.llm_provider == "weather_chain" else "OpenRouter"
    logger.info(
        f"Pipeline: Deepgram STT -> {_llm_label} LLM -> CosyVoice TTS -> MuseTalk avatar "
        f"(lang={config.language})"
    )

    stt = build_stt(config)
    # Memory harness: only for the weather bot, only when enabled. Wrapped around the
    # stateless chain (the chain can't hold memory); rewrites the query + distills the
    # conversation. Local qwen on CPU, so the GPU stays free for the avatar.
    memory = None
    if config.llm_provider == "weather_chain" and config.avatar_memory:
        memory = MemoryStore(
            base_dir=config.avatar_memory_dir,
            llm_url=config.memory_llm_url,
            llm_model=config.memory_llm_model,
            gated=config.memory_llm_gated,
            enabled=True,
        )
        logger.info(
            f"Avatar memory ON (model={config.memory_llm_model}, gated={config.memory_llm_gated})."
        )
        # Startup recovery: fold in any turns a crashed prior session left behind
        # (instant no-op on a normal boot). Runs before any client connects, so the
        # next conversation starts from a clean session.
        await memory.distill_pending()
    llm = build_llm(config, memory)
    tts = build_tts(config)
    avatar = build_avatar(config) if config.enable_avatar else None
    meter = TtfoMeter(target_s=config.ttfo_target_s)

    context = LLMContext([{"role": "system", "content": config.system_prompt}])
    # Two independent, optional tweaks to the user aggregator, both via LLMUserAggregatorParams:
    #   * Echo-guard (ECHO_GUARD=1): mute the mic while the bot speaks (half-duplex) via
    #     AlwaysUserMuteStrategy. BROKEN under steady sync (P11) -> default OFF.
    #   * No-interrupt (ALLOW_INTERRUPTIONS=0): the bot always finishes its turn; user speech
    #     during playback never cancels it. Done by turning OFF `enable_interruptions` on the
    #     default turn-START strategies (the flag that broadcasts the barge-in), keeping the
    #     default smart-turn STOP strategy. No mute state machine, so it's safe under steady.
    user_kwargs = {}
    if config.echo_guard:
        from pipecat.turns.user_mute import AlwaysUserMuteStrategy

        user_kwargs["user_mute_strategies"] = [AlwaysUserMuteStrategy()]
        logger.info("Echo-guard ON: mic muted while the bot speaks (half-duplex).")
    if not config.allow_interruptions:
        from pipecat.turns.user_start import (
            TranscriptionUserTurnStartStrategy,
            VADUserTurnStartStrategy,
        )
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        user_kwargs["user_turn_strategies"] = UserTurnStrategies(start=[
            VADUserTurnStartStrategy(enable_interruptions=False),
            TranscriptionUserTurnStartStrategy(enable_interruptions=False),
        ])
        logger.info("Interruptions OFF: the bot always finishes its turn (no barge-in).")
    user_params = None
    if user_kwargs:
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMUserAggregatorParams,
        )

        user_params = LLMUserAggregatorParams(**user_kwargs)
    aggregator = LLMContextAggregatorPair(context, user_params=user_params)

    stages = [
        transport.input(),       # mic in (+ VAD set in transport params)
        stt,                     # speech -> text
        aggregator.user(),       # add user turn to context
        llm,                     # text -> streamed text
        tts,                     # text -> streamed audio
    ]
    if avatar is not None:
        stages.append(avatar)    # audio -> lip-synced video+audio (server)
    stages += [
        meter,                   # measure TTFO
        transport.output(),      # -> browser (audio [+ video if avatar enabled])
        aggregator.assistant(),  # add bot turn to context
    ]
    pipeline = Pipeline(stages)

    _relax_bot_vad_stop_timeout()   # steady-mode screech fix (see the function's docstring)

    # Read-only transcript tap for the /nimbus/ chat bubbles (no pipeline structural change).
    _transcript = _TranscriptStore()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[_make_transcript_observer(_transcript)],
    )
    global _active_task, _active_transcript
    _active_task = task   # let the /client/measure-turn endpoint inject turns into this task
    _active_transcript = _transcript   # served by /client/transcript for the chat bubbles

    async def _warmup_llm():
        # Open the HTTPS connection to the LLM now, so the TLS handshake is done
        # before the user's first message (kills cold start). OpenRouter is
        # OpenAI-compatible, so the chat.completions warmup applies. The weather chain
        # has no cheap warmup ping (and no _client), so skip it there -- this warmup
        # would crash on the custom service otherwise.
        if config.llm_provider == "weather_chain":
            return
        model = getattr(getattr(llm, "_settings", None), "model", None)
        try:
            await llm._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            )
            logger.info("LLM connection pre-warmed.")
        except Exception as e:  # noqa: BLE001 — best-effort only
            logger.info(f"LLM warmup skipped: {e}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(transport, client):
        logger.info("Client connected — warming LLM + sending greeting.")
        asyncio.create_task(_warmup_llm())   # warm the LLM in the background
        if config.is_thai:
            greeting = "สวัสดีค่ะ พร้อมแล้วค่ะ พูดได้เลย"
        elif config.is_mandarin:
            greeting = "嗨，我準備好了，請說。"
            # Personalize from memory if we already know the returning user.
            if memory is not None:
                hint = memory.greeting_hint()
                if hint:
                    greeting = "嗨，歡迎回來！" + hint  # "Hi, welcome back! " + hint
        else:
            greeting = "Hi, I'm ready — go ahead."
        # Speak a fixed greeting directly via TTS (no LLM round-trip needed).
        await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(transport, client):
        global _active_task, _active_connection
        _active_task = None
        # Only release the single-connection slot if it's still ours -- a newer client
        # may have already claimed it (then this disconnect is us being kicked).
        if _active_connection is conn:
            _active_connection = None
        logger.info(f"Client disconnected. TTFO summary: {meter.summary()}")
        if memory is not None:
            try:
                await memory.distill_and_save()  # grow the human's memory after the chat
            except Exception as e:  # noqa: BLE001
                logger.warning(f"memory distill skipped ({type(e).__name__})")
        await task.cancel()

    await PipelineRunner().run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Entrypoint the Pipecat dev runner calls with a configured transport."""
    from pipecat.runner.utils import create_transport

    # A/V SYNC MODE -- this picks the transport's video clock, and the two modes are
    # MUTUALLY EXCLUSIVE in pipecat 1.3.0 (verified in base_output.py):
    #   * sync_with_audio  -> a tagged OutputImageRawFrame is routed through the AUDIO
    #     queue and only displayed after its preceding audio (per-frame A/V pinning).
    #     The transport renders it via `_video_images`, which is ONLY read when
    #     video_out_is_live is FALSE (the non-live `_video_task_handler` branch).
    #   * video_out_is_live -> frames go through an INDEPENDENT timed video queue on
    #     their own wall-clock; `_video_images` (and thus every sync_with_audio frame)
    #     is NEVER read. So with is_live=True the whole sync_with_audio mechanism is a
    #     no-op -- video plays on a free-running clock and drifts vs the voice.
    # MuseTalk emits video_start/video_clock/video_end markers and its client
    # (musetalk_video.py) buffers the voice + tags frames sync_with_audio for true lip
    # pinning, so the transport MUST be NON-live for that to work. We couple them off the
    # same flag: sync on (steady) -> is_live False (real sync); sync off (live) -> is_live
    # True (legacy free-running clock, animates but drifts). Idle frames are pushed
    # untagged either way and animate via `_set_video_image`.
    sync_av = config.avatar_sync_with_audio

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=config.enable_avatar,
            # See the A/V SYNC MODE note above: non-live so sync_with_audio actually
            # pins each frame to its audio. (is_live would silently disable the sync.)
            video_out_is_live=not sync_av,
            # Square portrait; MUST equal the avatar server's MUSETALK_SIZE and the
            # service's image_size (config.avatar_size couples all three off MUSETALK_SIZE,
            # same discipline as MUSETALK_FPS below). Smaller = far less WAN bandwidth.
            video_out_width=config.avatar_size,
            video_out_height=config.avatar_size,
            # MUST equal the rate the avatar server PUSHES frames, or playout starves/
            # piles up and the face drifts behind the audio (the "laggy/desynced" drift,
            # then a freeze). The server pumps frames at config.avatar_fps (MuseTalk ~20 --
            # it frame-drops to that rate so a sub-realtime GPU stays realtime), so this
            # MUST track the same value. Coupled here (and in avatar.py) so they can never
            # diverge again.
            video_out_framerate=max(1, round(config.avatar_fps)),
            vad_analyzer=build_vad_params(),
        ),
    }
    # Single-connection policy: a fresh offer kicks the previous session BEFORE we build
    # this one, so the single-client avatar server (:8002) is released before the new
    # pipeline reaches for it (two live sessions fight over the one shared GPU).
    conn = getattr(runner_args, "webrtc_connection", None)
    global _active_connection
    old = _active_connection
    _active_connection = conn   # claim the slot first so the old session's disconnect handler won't clear it
    if old is not None and old is not conn:
        logger.info("New WebRTC offer -- disconnecting the previous session (single-connection policy).")
        try:
            await old.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Kicking the previous session failed: {e!r}")

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, conn)


def _relax_bot_vad_stop_timeout() -> None:
    """THE steady-mode 'screech' root-cause fix (proven 2026-06-22).

    pipecat's output transport (`MediaSender._next_frame`) fires `_bot_stopped_speaking()` if no
    audio frame reaches its queue within `BOT_VAD_STOP_FALLBACK_SECS` (default **3s**) -- and that
    handler does `self._audio_buffer = bytearray()`, DISCARDING whatever partial audio is buffered.

    In `steady`/non-live sync the voice is held and released PACED TO RENDERED VIDEO frames. On a
    long reply the shared GPU render can stall > 3s, which starves the transport's audio queue, so
    the 3s timeout fires MID-TURN and discards the partial `_audio_buffer`. That discarded chunk is
    an arbitrary (usually ODD) byte count, so the remaining int16 PCM stream is left misaligned by
    an odd byte -> every subsequent sample straddles two real samples -> loud broadband noise to the
    end of the turn (the "screech"). Proven by byte-diffing the captures: a 1049-byte deletion at
    6.040s, speech otherwise bit-identical. `live` never hits this (it forwards audio continuously,
    so the queue is never starved 3s).

    We already drive an explicit `TTSStoppedFrame` per turn (`push_stop_frames=True` on the TTS
    service), which signals bot-stopped on its own, so the 3s audio-gap fallback is redundant here.
    Raise it so a render stall can never trigger the destructive discard -- a stall now just pauses
    the voice (steady's accepted behaviour) and it resumes CONTIGUOUS (clean), instead of screeching.

    The constant is read as a module global at `_next_frame()` call time (once per session, when the
    audio task starts), so patching the module attribute before the client connects takes effect.
    Knob: `BOT_VAD_STOP_FALLBACK_SECS` (seconds; <=0 leaves pipecat's 3s default)."""
    try:
        secs = float(os.getenv("BOT_VAD_STOP_FALLBACK_SECS", "600") or "600")
    except ValueError:
        secs = 600.0
    if secs <= 0:
        return
    try:
        from pipecat.transports import base_output
        base_output.BOT_VAD_STOP_FALLBACK_SECS = secs
        logger.info(f"BOT_VAD_STOP_FALLBACK_SECS -> {secs:g}s (steady-mode screech fix: a render "
                    f"stall can no longer discard the partial audio buffer mid-turn).")
    except Exception as e:  # noqa: BLE001 -- never block startup on this
        logger.warning(f"Could not relax BOT_VAD_STOP_FALLBACK_SECS: {e!r}")


def _configure_webrtc_video_bitrate() -> None:
    """Bound aiortc's VP8 send bitrate so the video stream FITS a remote/WAN link and
    can't starve it (the real cause of the "avatar trails the voice" stutter over the
    Thailand->Taiwan path).

    Why this is needed: pipecat's SmallWebRTCTransport hands raw frames to aiortc's VP8
    encoder, whose module-level limits are DEFAULT=500k, MIN=250k, MAX=1.5M (aiortc/codecs/
    vpx.py). It adapts DOWNWARD via REMB feedback, but (a) the 1.5M ceiling can overshoot a
    jittery consumer link -> packets queue -> the video falls progressively behind, and (b)
    the 250k floor can't absorb a worse dip -> loss -> freeze. pipecat's video_out_bitrate
    param is deprecated and wired to nothing, so the only place to set this is the aiortc
    module globals -- patched BEFORE the first encoder is created (it reads DEFAULT_BITRATE at
    init and the target_bitrate setter clamps to MIN/MAX). This keeps REMB's downward
    adaptation while capping the ceiling and lowering the floor (graceful degrade, no freeze).

    Knobs (bits/sec): WEBRTC_VIDEO_BITRATE (start point), _MAX (ceiling), _MIN (floor).
    Defaults suit a ~320px avatar over a multi-Mbps link; set _MAX=0 to leave aiortc as-is."""
    try:
        cap = int(os.getenv("WEBRTC_VIDEO_BITRATE_MAX", "600000") or "600000")
    except ValueError:
        cap = 600000
    if cap <= 0:
        return
    try:
        default = int(os.getenv("WEBRTC_VIDEO_BITRATE", "500000") or "500000")
        floor = int(os.getenv("WEBRTC_VIDEO_BITRATE_MIN", "120000") or "120000")
    except ValueError:
        default, floor = 500000, 120000
    try:
        from aiortc.codecs import vpx
    except Exception as e:  # noqa: BLE001
        logger.warning(f"WebRTC video bitrate config skipped (aiortc import: {e!r}).")
        return
    # Order matters: clamp default into the new [floor, cap] band.
    vpx.MIN_BITRATE = floor
    vpx.MAX_BITRATE = cap
    vpx.DEFAULT_BITRATE = max(floor, min(default, cap))
    logger.info(
        f"WebRTC VP8 bitrate bounded: min={floor} default={vpx.DEFAULT_BITRATE} max={cap} "
        f"(WEBRTC_VIDEO_BITRATE_MAX=0 to disable)."
    )


# All <head> patches for the served /client page collect here and ONE middleware injects
# them all. Why a shared list: each patch as its own middleware would race to serve the
# index (the outermost one wins and the others' patches silently vanish); a single
# serve-point keeps every env-gated patch additive and the prebuilt bundle untouched.
_client_head_patches: list[str] = []
_client_patch_middleware_installed = False
# Set by run_bot so the /client/measure-turn endpoint (the Measure button) can inject a turn
# into the live pipeline. None between sessions.
_active_task = None

# Single-connection policy: the current live WebRTC connection. The avatar server is
# single-client and two sessions fight over the one shared GPU, so a new client kicks the
# previous one -- bot() disconnects this when a fresh offer arrives. None between sessions.
_active_connection = None

# Live conversation transcript for the custom /nimbus/ chat bubbles. The pipeline has no RTVI
# processor in this build, so instead of a data channel we tap frames with a READ-ONLY observer
# (no pipeline structural change) into a small ring buffer the client polls via /client/transcript.
# Set per session by run_bot; None between sessions.
_active_transcript = None


class _TranscriptStore:
    """Append-only ring buffer of {seq, role, text} the /client/transcript endpoint serves.

    role is 'user' (a committed STT transcription) or 'bot' (the assistant's aggregated reply text). seq lets the
    client poll incrementally (?since=N). Typed /say turns are echoed client-side already and never
    produce a TranscriptionFrame, so they are not double-added here.
    """

    def __init__(self, cap: int = 200):
        self._items: list[dict] = []
        self._seq = 0
        self._cap = cap
        # The in-progress user utterance (STT interim results). Not seq'd -- it's a single
        # slot the client renders as one live bubble that updates in place, then is cleared
        # when the finalized TranscriptionFrame commits. See /client/transcript.
        self._partial: dict | None = None

    def add(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._seq += 1
        self._items.append({"seq": self._seq, "role": role, "text": text})
        if len(self._items) > self._cap:
            self._items = self._items[-self._cap:]

    def since(self, seq: int) -> list[dict]:
        return [it for it in self._items if it["seq"] > seq]

    def set_partial(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._partial = {"text": text, "updatedAt": time.time()}

    def clear_partial(self) -> None:
        self._partial = None

    @property
    def partial(self) -> dict | None:
        return self._partial


def _make_transcript_observer(store: "_TranscriptStore"):
    """A BaseObserver that records one user bubble per turn + the bot's aggregated reply text.

    Bot text arrives as a stream of LLMTextFrame tokens bracketed by LLMFullResponseStart/End;
    we accumulate between them and commit one 'bot' entry per reply. User STT arrives as
    InterimTranscriptionFrames (the live bubble) then one-or-more finalized TranscriptionFrames
    (one per speech pause); we accumulate the whole turn and commit ONE 'user' entry when the
    bot begins replying (LLMFullResponseStart). This only READS frames.
    """
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.frames.frames import (
        TranscriptionFrame,
        InterimTranscriptionFrame,
        LLMTextFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    )

    # No space between CJK segments (a space reads as a break mid-sentence); space for word langs.
    sep = "" if (config.is_mandarin or config.is_thai) else " "

    class _TranscriptObserver(BaseObserver):
        def __init__(self):
            super().__init__()
            self._buf = ""    # bot reply, accumulated between LLMFullResponseStart/End
            self._user = ""   # user turn, accumulated across STT segments until the bot replies
            self._seen = set()  # dedupe: observers can see a frame pushed by multiple processors

        async def on_push_frame(self, data):
            frame = data.frame
            fid = id(frame)
            if fid in self._seen:
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                # The bot starting to reply means the user's turn is complete: commit the WHOLE
                # accumulated turn as ONE bubble. Deepgram emits a TranscriptionFrame per speech
                # pause, so committing per-frame produced a bubble per pause ("a lot of bubbles").
                if self._user:
                    store.add("user", self._user)
                    self._user = ""
                store.clear_partial()  # the live bubble swaps for the committed one
                self._buf = ""
                self._seen.add(fid)
            elif isinstance(frame, LLMTextFrame):
                self._buf += frame.text or ""
                self._seen.add(fid)
            elif isinstance(frame, LLMFullResponseEndFrame):
                store.add("bot", self._buf)
                self._buf = ""
                store.clear_partial()  # backstop: drop any partial that never got a bot reply
                self._seen.add(fid)
            elif isinstance(frame, InterimTranscriptionFrame):
                # In-progress segment: show finalized-so-far + this live interim in the bubble.
                interim = (frame.text or "").strip()
                live = (self._user + sep + interim).strip() if self._user else interim
                store.set_partial(live)
                self._seen.add(fid)
            elif isinstance(frame, TranscriptionFrame):
                # A finalized STT segment -- accumulate; the single bubble commits at turn end
                # (LLMFullResponseStart above), not here.
                text = (frame.text or "").strip()
                if text:
                    self._user = (self._user + sep + text).strip() if self._user else text
                    store.set_partial(self._user)
                self._seen.add(fid)

    return _TranscriptObserver()


def _ensure_client_patch_middleware() -> bool:
    """Install (once) the middleware that serves /client with every registered head patch."""
    global _client_patch_middleware_installed
    if _client_patch_middleware_installed:
        return True
    try:
        from pathlib import Path as _Path

        import pipecat_ai_prebuilt
        from fastapi.responses import HTMLResponse

        from pipecat.runner.run import app
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Client page patch middleware skipped (import: {e!r}).")
        return False
    index_path = _Path(pipecat_ai_prebuilt.__file__).parent / "client" / "dist" / "index.html"
    if not index_path.is_file():
        logger.warning(f"Client page patch middleware skipped (no index.html at {index_path}).")
        return False

    @app.middleware("http")
    async def _inject_client_patches(request, call_next):
        # Debug beacon: injected client scripts POST what they observed on the device
        # (UA, output devices, which route path fired) so a phone problem is diagnosable
        # from pipeline.log instead of guessing what a remote browser did.
        if request.method == "POST" and request.url.path == "/client/speaker-debug":
            try:
                # Pipecat's runner logger.remove()'s every sink at startup and the file sink
                # only returns when a bot session starts (log_setup docstring) -- but the
                # page-load beacon arrives BEFORE any connect. Re-assert the sink so no
                # beacon is ever swallowed.
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                body = (await request.body())[:2000]
                logger.info(f"[speaker-debug] {body.decode('utf-8', 'replace')}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[speaker-debug] unreadable body: {e!r}")
            return HTMLResponse("", status_code=204)
        # Freeze beacon: the browser reports when the DISPLAYED avatar video stalls (the leg the
        # server/pipeline logs can't see). Logged as WARNING so a real freeze stands out.
        if request.method == "POST" and request.url.path == "/client/video-stall":
            try:
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                body = (await request.body())[:2000]
                logger.warning(f"[video-stall] {body.decode('utf-8', 'replace')}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[video-stall] unreadable body: {e!r}")
            return HTMLResponse("", status_code=204)
        # On-screen truth beacon: the browser's OWN WebRTC getStats() -- the DISPLAYED fps,
        # dropped/frozen frames, audio concealment, and the real played A/V skew (audio vs video
        # estimatedPlayoutTimestamp). This is what the user actually sees/hears, downstream of
        # everything the server can measure. INFO normally; WARNING when the sample carries a
        # glitch (g:1 -> dropped/frozen frames or audio concealment) so it stands out.
        if request.method == "POST" and request.url.path == "/client/av-stats":
            try:
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                body = (await request.body())[:2000]
                text = body.decode("utf-8", "replace")
                if '"g":1' in text:
                    logger.warning(f"[av-stats] {text}")
                else:
                    logger.info(f"[av-stats] {text}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[av-stats] unreadable body: {e!r}")
            return HTMLResponse("", status_code=204)
        # Playout beacon: the browser reports the instant the bot's VOICE actually starts
        # playing (to the ear) -- the last mile the server clock can't see. measure.py stitches
        # its epoch onto the log t0 to close the waterfall. See _install_client_playout_probe.
        if request.method == "POST" and request.url.path == "/client/playout":
            try:
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                body = (await request.body())[:2000]
                logger.info(f"[client-playout] {body.decode('utf-8', 'replace')}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[client-playout] unreadable body: {e!r}")
            return HTMLResponse("", status_code=204)
        # Nimbus transcript poll: the /nimbus/ chat polls this for new conversation lines (the bot's
        # spoken reply text + finalized user speech), captured by the read-only transcript observer.
        # ?since=<seq> returns only newer entries, plus "partial" = the in-progress user
        # utterance (STT interim, {"text","updatedAt"} | null) rendered as one live bubble.
        # JSON: {"items":[{"seq","role","text"}, ...], "partial": {...} | null}.
        if request.method == "GET" and request.url.path == "/client/transcript":
            import json as _json
            try:
                since = int(request.query_params.get("since", "0"))
            except (TypeError, ValueError):
                since = 0
            items = _active_transcript.since(since) if _active_transcript is not None else []
            partial = _active_transcript.partial if _active_transcript is not None else None
            return HTMLResponse(
                _json.dumps({"items": items, "partial": partial}),
                media_type="application/json",
            )
        # Nimbus text send: inject a TYPED user turn (from the /nimbus/ chat box) into the live
        # pipeline as a real user message -> LLM -> TTS -> avatar speaks it. Voice-first stays the
        # primary path; this is the keyboard alternative and reuses the same _active_task inject as
        # the measure button. Body: {"text": "..."}.
        if request.method == "POST" and request.url.path == "/client/say":
            try:
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                from pipecat.frames.frames import LLMMessagesAppendFrame

                import json as _json
                raw = (await request.body())[:4000]
                text = (_json.loads(raw or b"{}").get("text") or "").strip()
                if not text:
                    return HTMLResponse("empty", status_code=400)
                if _active_task is None:
                    logger.warning("[say] no active session (client not connected?)")
                    return HTMLResponse("no active session", status_code=409)
                await _active_task.queue_frames([
                    LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True)
                ])
                logger.info(f"[say] injected typed turn: {text!r}")
                return HTMLResponse("", status_code=204)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[say] failed: {e!r}")
                return HTMLResponse("error", status_code=500)
        # Measure button: inject a real bot turn on demand (a fixed question through the full
        # LLM->TTS->avatar path) so the browser can time click -> voice-onset WITHOUT depending on
        # mic/VAD/STT turn-taking (which logs no [TTFO for real browser turns). See
        # _install_measure_button.
        if request.method == "POST" and request.url.path == "/client/measure-turn":
            try:
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                from pipecat.frames.frames import LLMMessagesAppendFrame

                q = {"zh": "什麼是人工智慧？請用一句話簡短回答。",
                     "th": "AI คืออะไร ตอบสั้น ๆ หนึ่งประโยค",
                     "en": "What is AI? Answer in one short sentence."}.get(
                    config.language, "What is AI? Answer in one short sentence.")
                if _active_task is None:
                    logger.warning("[measure-turn] no active session (client not connected?)")
                    return HTMLResponse("no active session", status_code=409)
                await _active_task.queue_frames([
                    LLMMessagesAppendFrame(messages=[{"role": "user", "content": q}], run_llm=True)
                ])
                logger.info(f"[measure-turn] injected turn: {q!r}")
                return HTMLResponse("", status_code=204)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[measure-turn] failed: {e!r}")
                return HTMLResponse("error", status_code=500)
        # Only the index page (exact /client or /client/); assets pass through to the mount.
        if request.method == "GET" and request.url.path in ("/client", "/client/"):
            try:
                html = index_path.read_text(encoding="utf-8").replace(
                    "<head>", "<head>" + "".join(_client_head_patches), 1
                )
                # no-store: a phone that cached the pre-patch index would silently miss
                # every injected fix (bit us 2026-07-04); the page is tiny, always refetch.
                return HTMLResponse(html, headers={"Cache-Control": "no-store"})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Client page patch failed; serving default page: {e!r}")
        return await call_next(request)

    _client_patch_middleware_installed = True
    return True


def _install_client_jitter_buffer() -> None:
    """Inject a receive-side WebRTC jitter buffer into the served /client page so EVERY
    device that opens it absorbs network jitter (a smoother avatar over a remote/WAN/Tailscale
    link) with no per-device console tweak.

    Why this exists: over a remote link the lag is the NETWORK (jitter), not the render. The
    standard fix is a bigger receive-side jitter buffer -- it holds a few hundred ms so late
    packets still arrive in time (smoother, at the cost of that much added latency). The browser
    has one but its default target is small; this raises it.

    How: pure HTML injection -- the prebuilt bundle is untouched. A synchronous <script> in
    <head> runs before the deferred ES-module bundle, so it patches window.RTCPeerConnection
    first; every media receiver created then gets jitterBufferTarget (Chromium) / playoutDelayHint
    (legacy) set. Configurable via CLIENT_JITTER_BUFFER_MS (default 400; 0 disables)."""
    try:
        ms = int(os.getenv("CLIENT_JITTER_BUFFER_MS", "400") or "400")
    except ValueError:
        ms = 400
    if ms <= 0:
        return
    # Minified inline patch: wrap RTCPeerConnection so each receiver requests a `ms` buffer.
    patch = (
        "<script>(()=>{const T=" + str(ms) + ";const N=window.RTCPeerConnection;"
        "if(!N||N.__jb)return;const P=function(...a){const pc=new N(...a);"
        "pc.addEventListener('track',e=>{try{const r=e.receiver;"
        "if('jitterBufferTarget' in r)r.jitterBufferTarget=T;"
        "else if('playoutDelayHint' in r)r.playoutDelayHint=T/1000;}catch(_){}});"
        "return pc;};P.prototype=N.prototype;P.__jb=1;window.RTCPeerConnection=P;"
        "console.log('[jitter-buffer] receiver target '+T+'ms');})();</script>"
    )
    if not _ensure_client_patch_middleware():
        return
    _client_head_patches.append(patch)
    logger.info(f"Client jitter buffer ENABLED: {ms}ms (CLIENT_JITTER_BUFFER_MS=0 to disable).")


def _install_client_speaker_route() -> None:
    """On a PHONE browser, route the bot's voice to the LOUDSPEAKER instead of the earpiece.

    Why this exists: when a WebRTC page holds a live mic, Android Chrome flips the phone into
    'communication' audio routing, which plays the remote track through the EARPIECE (quiet,
    hold-to-your-ear phone-call routing) -- the avatar's voice is near-inaudible on a phone
    lying on a desk. Fix = the Audio Output Devices API: find the 'Speakerphone/Speaker'
    audiooutput device and setSinkId() every media element to it.

    How (two routes, tried in order per media element -- hooked at
    HTMLMediaElement.prototype.play so it also catches elements the bundle never attaches
    to the DOM or hides in a shadow root, which a querySelector sweep misses):
      1. setSinkId to the 'Speakerphone/Speaker' audiooutput device (Android Chrome).
         Labels are empty until the mic permission is granted, so the pick re-runs on
         each play() and on 'devicechange' (headset plug/unplug re-picks).
      2. WebAudio fallback (iOS Safari has no setSinkId/output labels): pipe the element's
         MediaStream through an AudioContext -- AudioContext output uses the media/playback
         route (loudspeaker), not the earpiece 'communication' route. The element keeps
         playing but muted (Safari needs a live sink on the stream or WebAudio goes silent),
         and it is only muted AFTER the context is confirmed running so audio can never
         disappear entirely; a one-shot pointerdown resume covers iOS's gesture rule.

    Scope/safety: mobile user agents ONLY -- a desktop user may be on headphones, and
    forcing the speaker sink there would yank audio away from them. Every step reports to
    POST /client/speaker-debug -> pipeline.log, so a remote phone is diagnosable.
    CLIENT_FORCE_SPEAKER=0 disables (default ON -- same default-on convention as the jitter
    buffer: it self-gates to the devices that need it)."""
    if (os.getenv("CLIENT_FORCE_SPEAKER", "1") or "1").lower() in ("0", "false", "no", "off"):
        return
    patch = (
        "<script>(()=>{if(!/Android|iPhone|iPad|Mobi/i.test(navigator.userAgent))return;"
        "const log=(...a)=>console.log('[speaker-route]',...a);"
        "const bea=o=>{try{fetch('/client/speaker-debug',{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})}catch(_){}};"
        "let want=null,ctx=null;"
        "async function pick(){try{const ds=await navigator.mediaDevices.enumerateDevices();"
        "const outs=ds.filter(d=>d.kind==='audiooutput');"
        "const s=outs.find(d=>/speaker/i.test(d.label||''));"
        "bea({ev:'pick',outs:outs.map(d=>d.label||'?'),chose:s?s.label:null});"
        "return s?s.deviceId:null}catch(e){bea({ev:'pick-err',err:String(e)});return null}}"
        "async function route(el){try{"
        "if(typeof el.setSinkId==='function'){"
        "if(want==null)want=await pick();"
        "if(want){if(el.sinkId!==want){await el.setSinkId(want);log('sink -> loudspeaker');"
        "bea({ev:'setSinkId-ok'})}return}}"
        "const ms=el.srcObject;"
        "if(!ms||!ms.getAudioTracks||!ms.getAudioTracks().length||el.__spk)return;"
        "el.__spk=1;ctx=ctx||new (window.AudioContext||window.webkitAudioContext)();"
        "ctx.createMediaStreamSource(ms).connect(ctx.destination);"
        "const fin=()=>{el.muted=true;el.volume=0;log('webaudio -> loudspeaker');"
        "bea({ev:'webaudio-ok',state:ctx.state})};"
        "if(ctx.state==='running')fin();"
        "else{ctx.resume().then(fin).catch(()=>{});"
        "document.addEventListener('pointerdown',()=>ctx.resume().then(fin).catch(()=>{}),{once:true})}"
        "}catch(e){bea({ev:'route-err',err:String(e&&e.message||e)})}}"
        "const P=HTMLMediaElement.prototype.play;"
        "HTMLMediaElement.prototype.play=function(){route(this);return P.apply(this,arguments)};"
        "if(navigator.mediaDevices&&navigator.mediaDevices.addEventListener)"
        "navigator.mediaDevices.addEventListener('devicechange',()=>{want=null});"
        "bea({ev:'loaded',ua:navigator.userAgent,"
        "sink:'setSinkId' in HTMLMediaElement.prototype});})();</script>"
    )
    if not _ensure_client_patch_middleware():
        return
    _client_head_patches.append(patch)
    logger.info("Client speaker route ENABLED: phone browsers play via the loudspeaker "
                "(CLIENT_FORCE_SPEAKER=0 to disable).")


def _install_client_video_stall_monitor() -> None:
    """Beacon the avatar's REAL displayed freeze (browser side) to pipeline.log.

    Why this exists: the server + pipeline logs only see up to the WebRTC send. A freeze in the
    transport or the browser's own decode/playout is invisible there -- the 2026-07-05 gap: a
    clean-render turn (server logs healthy, no dropped segments) still froze on screen, so the
    cause had to be downstream of everything logged. This attaches requestVideoFrameCallback to
    the avatar <video> and reports when the gap between DISPLAYED frames exceeds
    CLIENT_VIDEO_STALL_MS, so a freeze is captured wherever it actually lives, with the media
    time + UA to localize it. A poller also fires for a freeze still in progress (rVFC can't
    report its own stall). Pairs with the pipeline-side [avatar FREEZE] watchdog.

    How: the same sanctioned <head> injection as the jitter buffer + speaker route -- the
    prebuilt bundle is untouched, and play() is hooked so it catches the <video> whenever it
    starts (even if the bundle never attaches it to the DOM). CLIENT_VIDEO_STALL_MONITOR=0
    disables; CLIENT_VIDEO_STALL_MS sets the gap that counts as a freeze. Default OFF (diagnostic
    scaffolding); set CLIENT_VIDEO_STALL_MONITOR=1 to re-arm it when hunting a freeze."""
    if (os.getenv("CLIENT_VIDEO_STALL_MONITOR", "0") or "0") == "0":
        return
    try:
        thr = int(os.getenv("CLIENT_VIDEO_STALL_MS", "350") or "350")
    except ValueError:
        thr = 350
    patch = (
        "<script>(()=>{try{const T=" + str(thr) + ";"
        "const bea=o=>{try{fetch('/client/video-stall',{method:'POST',keepalive:true,"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});}catch(_){}};"
        "let last=0,lastMT=0,open=false,vid=null,n=0;"
        "const attach=v=>{if(!v||v.__vsm||!v.requestVideoFrameCallback)return;v.__vsm=1;vid=v;"
        "const cb=(now,meta)=>{if(last){const g=now-last;if(g>=T){n++;open=false;"
        "bea({ev:'stall',gap:Math.round(g),mediaT:+(meta.mediaTime||0).toFixed(2),n:n,"
        "ua:navigator.userAgent.slice(0,60)});}}last=now;lastMT=(meta&&meta.mediaTime)||0;"
        "try{v.requestVideoFrameCallback(cb);}catch(_){}};"
        "try{v.requestVideoFrameCallback(cb);}catch(_){}};"
        "const P=HTMLMediaElement.prototype.play;HTMLMediaElement.prototype.play=function(){"
        "if(this.tagName==='VIDEO')attach(this);return P.apply(this,arguments);};"
        "setInterval(()=>{if(vid&&last&&!open){const g=performance.now()-last;if(g>=T){open=true;"
        "bea({ev:'ongoing',gap:Math.round(g),mediaT:+lastMT.toFixed(2),"
        "ua:navigator.userAgent.slice(0,60)});}}},300);"
        "console.log('[video-stall-monitor] on (>'+T+'ms)');}catch(_){}})();</script>"
    )
    if not _ensure_client_patch_middleware():
        return
    _client_head_patches.append(patch)
    logger.info(f"Client video-stall monitor ENABLED: displayed-frame gaps >{thr}ms beaconed to "
                "pipeline.log (CLIENT_VIDEO_STALL_MONITOR=0 to disable).")


def _install_client_av_stats_monitor() -> None:
    """Beacon the browser's OWN measurement of what the user actually sees/hears.

    Why this exists: the pipeline's [musetalk sync] hold=/[avatar timing] lines measure what was
    QUEUED and SENT (server side); the rVFC monitor above catches only outright FREEZES. Neither
    tells you the sustained DISPLAYED fps, the frames the browser dropped/froze, audio glitches,
    or the real A/V skew as PLAYED -- exactly the "some voice delay / some lag" the eye reports
    but the server logs can't see. WebRTC's getStats() is the authoritative source: inbound-rtp
    gives framesPerSecond/framesDropped/freezeCount/totalFreezesDuration (video actually painted),
    concealedSamples/concealmentEvents (audio actually stretched/hidden = glitches), and
    estimatedPlayoutTimestamp on BOTH tracks -> their difference is the real played lip-sync
    offset (audio ahead of video => voice ahead of lips, the server's "LIPS BEHIND", now confirmed
    on the device). Sampled once per CLIENT_AV_STATS_MS as per-interval DELTAS, beaconed to
    pipeline.log; a sample with dropped/frozen frames or audio concealment is flagged g:1 (logged
    WARNING). Pairs with the freeze monitor + the pipeline-side [avatar FREEZE] watchdog.

    How: capture the RTCPeerConnection by hooking its prototype.setRemoteDescription (always
    called once during SDP exchange) to grab the instance -- the same "wrap a prototype method"
    idiom as the speaker route's play() hook, so the global constructor + its statics stay intact
    and the prebuilt bundle is untouched. CLIENT_AV_STATS_MONITOR=0 disables; CLIENT_AV_STATS_MS
    sets the sample interval."""
    # Default OFF (diagnostic scaffolding); set CLIENT_AV_STATS_MONITOR=1 to re-arm.
    if (os.getenv("CLIENT_AV_STATS_MONITOR", "0") or "0") == "0":
        return
    try:
        iv = int(os.getenv("CLIENT_AV_STATS_MS", "1000") or "1000")
    except ValueError:
        iv = 1000
    patch = (
        "<script>(()=>{try{const IV=" + str(iv) + ";"
        "const bea=o=>{try{fetch('/client/av-stats',{method:'POST',keepalive:true,"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});}catch(_){}};"
        "const PR=(window.RTCPeerConnection||window.webkitRTCPeerConnection||{}).prototype;"
        "if(!PR||!PR.getStats||!PR.setRemoteDescription)return;let pc=null;"
        "const SRD=PR.setRemoteDescription;"
        "PR.setRemoteDescription=function(){pc=this;return SRD.apply(this,arguments);};let p={};"
        "setInterval(async()=>{if(!pc)return;let s;try{s=await pc.getStats();}catch(_){return;}"
        "let v=null,a=null;s.forEach(r=>{if(r.type==='inbound-rtp'){"
        "if(r.kind==='video')v=r;else if(r.kind==='audio')a=r;}});if(!v&&!a)return;"
        "const now=performance.now(),o={};let act=false,g=false;"
        "if(v){const dt=(now-(p.t||now))/1000,df=(v.framesDecoded||0)-(p.vfd||0);"
        "o.fps=dt>0?+(df/dt).toFixed(1):0;if(df>0)act=true;"
        "o.drop=(v.framesDropped||0)-(p.vdr||0);o.frz=(v.freezeCount||0)-(p.vfz||0);"
        "o.frzS=+(((v.totalFreezesDuration||0)-(p.vfzs||0))).toFixed(2);"
        "o.pause=(v.pauseCount||0)-(p.vpa||0);if(v.framesPerSecond!=null)o.rfps=v.framesPerSecond;"
        "if(o.drop||o.frz)g=true;"
        "p.vfd=v.framesDecoded;p.vdr=v.framesDropped;p.vfz=v.freezeCount;"
        "p.vfzs=v.totalFreezesDuration;p.vpa=v.pauseCount;}"
        "if(a){o.aconc=(a.concealedSamples||0)-(p.ac||0);o.alost=(a.packetsLost||0)-(p.al||0);"
        "if(a.jitterBufferDelay!=null&&a.jitterBufferEmittedCount){"
        "const dd=a.jitterBufferDelay-(p.ajd||0),de=a.jitterBufferEmittedCount-(p.aje||0);"
        "o.ajbuf=de>0?+(dd/de).toFixed(3):0;}if(o.aconc>0||o.alost>0)g=true;"
        "p.ac=a.concealedSamples;p.al=a.packetsLost;p.ajd=a.jitterBufferDelay;"
        "p.aje=a.jitterBufferEmittedCount;}"
        "if(v&&a&&v.estimatedPlayoutTimestamp&&a.estimatedPlayoutTimestamp){"
        "o.avskew=+((a.estimatedPlayoutTimestamp-v.estimatedPlayoutTimestamp)/1000).toFixed(3);}"
        "p.t=now;if(g)o.g=1;if(act||g)bea(o);},IV);"
        "console.log('[av-stats-monitor] on (every '+IV+'ms)');}catch(_){}})();</script>"
    )
    if not _ensure_client_patch_middleware():
        return
    _client_head_patches.append(patch)
    logger.info(f"Client A/V-stats monitor ENABLED: displayed fps + drops/freezes + audio "
                f"concealment + played A/V skew sampled every {iv}ms to pipeline.log "
                "(CLIENT_AV_STATS_MONITOR=0 to disable).")


def _install_client_playout_probe() -> None:
    """Beacon the instant the bot's VOICE first plays in the browser (to the ear).

    Why: measure.py can clock the moment audio ARRIVES at a headless client, but the true
    to-the-ear moment adds the browser's own jitter buffer + decode + speaker route. This taps
    the actual played audio and reports its onset, so the latency waterfall can close the last
    mile with a real device instead of an estimate.

    How: hook HTMLMediaElement.prototype.play (the same prototype-method-hook idiom as the
    speaker route, so the prebuilt bundle is untouched and it catches elements a DOM sweep
    misses). On play, if the element's srcObject carries an audio track, pipe it through an
    AudioContext AnalyserNode and watch the RMS; the first frame above threshold beacons
    {"ev":"audio-onset","t":Date.now()} to /client/playout, then re-arms after ~0.5s of silence
    for the next turn. Default OFF (measurement scaffolding); CLIENT_PLAYOUT_PROBE=1 to arm."""
    if (os.getenv("CLIENT_PLAYOUT_PROBE", "0") or "0") == "0":
        return
    patch = (
        "<script>(()=>{try{"
        "const bea=o=>{try{fetch('/client/playout',{method:'POST',keepalive:true,"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});}catch(_){}};"
        "let ctx=null,armed=true,quiet=0;const seen=new WeakSet();"
        "const watch=stream=>{try{ctx=ctx||new (window.AudioContext||window.webkitAudioContext)();"
        "const src=ctx.createMediaStreamSource(stream);const an=ctx.createAnalyser();"
        "an.fftSize=512;const buf=new Float32Array(an.fftSize);src.connect(an);"
        "const tick=()=>{an.getFloatTimeDomainData(buf);let s=0;"
        "for(let i=0;i<buf.length;i++)s+=buf[i]*buf[i];const rms=Math.sqrt(s/buf.length);"
        "if(rms>0.02){if(armed){armed=false;bea({ev:'audio-onset',t:Date.now()});}quiet=0;}"
        "else if(!armed){if(++quiet>30)armed=true;}"
        "requestAnimationFrame(tick);};tick();"
        "console.log('[playout-probe] watching bot audio');}catch(_){}};"
        "const M=HTMLMediaElement.prototype,PL=M.play;"
        "M.play=function(){try{const st=this.srcObject;"
        "if(st&&st.getAudioTracks&&st.getAudioTracks().length&&!seen.has(st)){"
        "seen.add(st);watch(st);}}catch(_){}return PL.apply(this,arguments);};"
        "console.log('[playout-probe] armed');}catch(_){}})();</script>"
    )
    if not _ensure_client_patch_middleware():
        return
    _client_head_patches.append(patch)
    logger.info("Client playout probe ENABLED: first-voice-onset beaconed to /client/playout "
                "(CLIENT_PLAYOUT_PROBE=0 to disable).")


def _install_measure_button() -> None:
    """A 'Measure turn' button (MEASURE_BUTTON=1, default OFF) — the reliable way to get the real
    to-the-ear latency without fighting mic/VAD/STT turn-taking or a passive audio hook. On click
    (a USER GESTURE, which lets us resume the AudioContext — the passive playout probe can't
    guarantee that, the likely reason it never fired) it taps the played bot audio, fires ONE real
    turn via POST /client/measure-turn, times click -> first-voice-onset and shows it in-page, and
    beacons the onset to /client/playout so `measure.py --from-browser` fills the last-mile row."""
    if (os.getenv("MEASURE_BUTTON", "0") or "0") == "0":
        return
    patch = (
        "<script>(()=>{try{"
        "const mk=()=>{if(document.getElementById('measBtn'))return;"
        "const b=document.createElement('button');b.id='measBtn';b.textContent='Measure turn';"
        "b.style.cssText='position:fixed;z-index:99999;left:12px;bottom:12px;padding:10px 14px;"
        "font:600 14px system-ui;background:#111;color:#fff;border:1px solid #555;border-radius:8px;cursor:pointer';"
        "const o=document.createElement('div');o.id='measOut';o.style.cssText='position:fixed;z-index:99999;"
        "left:12px;bottom:58px;font:600 13px system-ui;color:#6f6;background:#000c;padding:6px 10px;"
        "border-radius:6px;display:none;max-width:70vw';document.body.appendChild(b);document.body.appendChild(o);"
        "let ctx=null,an=null;"
        "const findS=()=>{for(const e of document.querySelectorAll('audio,video')){const s=e.srcObject;"
        "if(s&&s.getAudioTracks&&s.getAudioTracks().length)return s;}return null;};"
        "b.onclick=async()=>{const s=findS();"
        "if(!s){o.style.display='block';o.textContent='Connect + allow audio first.';return;}"
        "try{ctx=ctx||new(window.AudioContext||window.webkitAudioContext)();await ctx.resume();}catch(_){}"
        "if(!an){try{const src=ctx.createMediaStreamSource(s);an=ctx.createAnalyser();an.fftSize=512;src.connect(an);}catch(_){}}"
        "if(!an){o.style.display='block';o.textContent='cannot tap audio';return;}"
        "const buf=new Float32Array(an.fftSize);o.style.display='block';o.textContent='measuring...';b.disabled=true;"
        "const t0=Date.now();let armed=true;try{await fetch('/client/measure-turn',{method:'POST'});}catch(_){}"
        "const dl=t0+15000;const tick=()=>{an.getFloatTimeDomainData(buf);let m=0;"
        "for(let i=0;i<buf.length;i++)m+=buf[i]*buf[i];const rms=Math.sqrt(m/buf.length);"
        "if(rms>0.02&&armed){armed=false;const t1=Date.now();o.textContent='heard '+(t1-t0)+' ms after click';"
        "b.disabled=false;try{fetch('/client/playout',{method:'POST',keepalive:true,"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({ev:'audio-onset',t:t1,src:'button'})});}catch(_){}return;}"
        "if(Date.now()<dl)requestAnimationFrame(tick);else{o.textContent='no audio in 15s';b.disabled=false;}};tick();}};"
        "if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mk);else mk();"
        "setInterval(mk,2000);console.log('[measure-button] installed');}catch(_){}})();</script>"
    )
    if not _ensure_client_patch_middleware():
        return
    _client_head_patches.append(patch)
    logger.info("Measure button ENABLED: click 'Measure turn' to fire+time a turn "
                "(MEASURE_BUTTON=0 to disable).")


def _restrict_ice_to_subnet() -> None:
    """Restrict WebRTC host candidates to ONE network (default: the Tailscale CGNAT range
    100.64.0.0/10), so ICE only ever offers the interface that can actually reach a remote
    tailnet viewer.

    Why this is needed (root cause of the intermittent mic, 2026-06-21): this box has
    several adapters -- Tailscale (100.x), Hyper-V (172.x), Radmin/Hamachi (26.x), LAN
    (192.168.x) -- and aiortc/aioice gather a host candidate for EVERY one. ICE then checks
    a large matrix of pairs, but only the Tailscale pair (100.x <-> the remote's 100.x) can
    actually reach a remote tailnet peer; the rest are dead. Worse, a marginal pair can win
    nomination and then drop ('Consent to send expired' in the logs) -> the audio track
    errors ('Media stream error; clearing track' / recv None) -> the mic dies mid-call. That
    is the "works sometimes, mostly not" symptom. The Tailscale pair is VERIFIED reachable in
    the logs (State.IN), so pinning ICE to it makes the stable path win immediately -- no
    relay/TURN needed.

    Patches aioice.ice.get_host_addresses (the host-candidate source) BEFORE the runner
    builds any peer connection, same module-global approach as the bitrate cap above. Safe by
    construction: WEBRTC_ICE_SUBNET=0 (or empty) disables it, and if the filter would drop
    EVERY address (e.g. Tailscale is down) it falls back to the full list so a local/LAN
    connection still works. A local browser can still reach the 100.x interface, so same-box
    testing is unaffected."""
    import ipaddress

    subnet_str = os.getenv("WEBRTC_ICE_SUBNET", "100.64.0.0/10")
    if not subnet_str or subnet_str == "0":
        return
    try:
        net = ipaddress.ip_network(subnet_str, strict=False)
    except ValueError:
        logger.warning(f"WEBRTC_ICE_SUBNET={subnet_str!r} invalid; ICE restriction skipped.")
        return
    try:
        from aioice import ice as _ice
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ICE interface restriction skipped (aioice import: {e!r}).")
        return

    _orig = _ice.get_host_addresses

    def _filtered(use_ipv4: bool, use_ipv6: bool):
        addrs = _orig(use_ipv4, use_ipv6)
        kept = []
        for a in addrs:
            try:
                if ipaddress.ip_address(a) in net:
                    kept.append(a)
            except ValueError:
                continue  # skip anything not a plain IP (e.g. scoped IPv6)
        if not kept:
            logger.warning(
                f"No host address in {subnet_str} (Tailscale down?); keeping all "
                f"{len(addrs)} addresses so the connection still works."
            )
            return addrs
        return kept

    _ice.get_host_addresses = _filtered
    logger.info(
        f"WebRTC ICE host candidates restricted to {subnet_str} "
        f"(was {len(_orig(True, False))} v4 addrs; WEBRTC_ICE_SUBNET=0 to disable)."
    )


def _configure_stun_servers() -> None:
    """Inject STUN servers into SmallWebRTCRequestHandler so the server discovers
    its public IP as a server-reflexive ICE candidate.

    Needed on GCP (and any 1:1-NAT host) where the public IP is not assigned to any
    local interface — without STUN the server only advertises its Docker/private IP
    and the browser cannot reach it via UDP.  WEBRTC_STUN_URL=0 disables."""
    stun_url = os.getenv("WEBRTC_STUN_URL", "stun:stun.l.google.com:19302")
    if not stun_url or stun_url == "0":
        return
    try:
        from pipecat.transports.smallwebrtc.connection import IceServer
        from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler

        _orig_init = SmallWebRTCRequestHandler.__init__

        def _patched_init(self, **kwargs):
            _orig_init(self, **kwargs)
            self._ice_servers = [IceServer(urls=stun_url)]

        SmallWebRTCRequestHandler.__init__ = _patched_init
        logger.info(f"WebRTC STUN configured: {stun_url} (WEBRTC_STUN_URL=0 to disable).")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"STUN setup skipped: {e!r}")


def _install_nimbus_client() -> None:
    """Serve the custom 'Nimbus AI' UI at /nimbus/ (the figma-to-code redesign).

    A self-contained vanilla-JS client (no build step) that speaks the SAME
    SmallWebRTC signaling as the prebuilt bundle -- POST /api/offer, then the
    avatar video + bot audio arrive as WebRTC tracks and the mic goes up the same
    connection. This is ADDITIVE: the prebuilt bundle at /client is untouched and
    stays the fallback. Mounted as StaticFiles so index.html + presenter.png serve
    from one dir; served no-store so a phone never caches a stale build.
    """
    from pathlib import Path as _Path

    client_dir = _Path(__file__).resolve().parent.parent / "local_services" / "nimbus_client"
    if not (client_dir / "index.html").is_file():
        logger.warning(f"Nimbus client not mounted (no index.html at {client_dir}).")
        return
    try:
        from starlette.staticfiles import StaticFiles
        from pipecat.runner.run import app
    except Exception as e:  # pragma: no cover - only when runner app isn't importable
        logger.warning(f"Nimbus client mount skipped ({e!r}).")
        return

    class _NoStoreStatic(StaticFiles):
        def is_not_modified(self, *a, **k):
            return False  # never 304 -> the phone always gets the latest build

        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-store"
            return resp

    app.mount("/nimbus", _NoStoreStatic(directory=str(client_dir), html=True), name="nimbus")
    logger.info("Nimbus UI mounted at /nimbus/ (custom client; /client prebuilt untouched).")


if __name__ == "__main__":
    import sys

    # Windows consoles default to cp1252; Pipecat's runner prints emoji. Force
    # UTF-8 so startup doesn't crash on the banner.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # Durable per-process log at logs/pipeline.log (rotated, full tracebacks,
    # plus uvicorn/asyncio via the stdlib intercept). See log_setup.py.
    from log_setup import setup_logging

    setup_logging("pipeline")

    # Bound the VP8 send bitrate so the video fits a remote/WAN link (no starvation/freeze)
    # BEFORE any peer connection is built, then serve the prebuilt UI at /client with a
    # receive-side jitter buffer injected so every device (esp. remote/Tailscale viewers)
    # gets smoother playback automatically.
    _configure_webrtc_video_bitrate()
    _install_client_jitter_buffer()
    # Phone browsers: play the bot's voice on the loudspeaker, not the earpiece (Android
    # Chrome's live-mic 'communication' routing). Env-gated; bundle untouched.
    _install_client_speaker_route()
    # Capture the avatar's REAL displayed freeze (browser side) -> pipeline.log; the server logs
    # only see up to the WebRTC send. Env-gated; bundle untouched.
    _install_client_video_stall_monitor()
    # Beacon the browser's OWN getStats() -> the DISPLAYED fps, dropped/frozen frames, audio
    # concealment, and the real played A/V skew: what the user actually sees/hears. Env-gated.
    _install_client_av_stats_monitor()
    # Real-browser voice-onset beacon (to-the-ear last mile for the measure waterfall).
    _install_client_playout_probe()
    # On-demand 'Measure turn' button: fire a real turn on a click + time click->voice-onset.
    _install_measure_button()
    # Serve the custom 'Nimbus AI' redesign at /nimbus/ (additive; /client stays the fallback).
    _install_nimbus_client()
    # Pin ICE host candidates to the Tailscale interface so the stable 100.x<->100.x pair
    # wins immediately (kills the intermittent-mic ICE pollution -- see the function docstring).
    _restrict_ice_to_subnet()
    # Add STUN so the server discovers its public IP on GCP / any 1:1-NAT host.
    _configure_stun_servers()
    from pipecat.runner.run import main

    main()
