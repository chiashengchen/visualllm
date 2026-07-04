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


async def run_bot(transport: BaseTransport) -> None:
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
    # Echo-guard: mute the mic while the bot is speaking (half-duplex) so the
    # avatar's own voice leaking into the mic can't trigger a barge-in that wipes
    # the in-flight render mid-turn (the self-interruption seen in the logs). Uses
    # Pipecat's built-in AlwaysUserMuteStrategy. ECHO_GUARD=0 restores barge-in.
    user_params = None
    if config.echo_guard:
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMUserAggregatorParams,
        )
        from pipecat.turns.user_mute import AlwaysUserMuteStrategy

        user_params = LLMUserAggregatorParams(
            user_mute_strategies=[AlwaysUserMuteStrategy()]
        )
        logger.info("Echo-guard ON: mic muted while the bot speaks (half-duplex).")
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

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
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


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
    try:
        from pathlib import Path as _Path

        import pipecat_ai_prebuilt
        from fastapi.responses import HTMLResponse

        from pipecat.runner.run import app
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Client jitter-buffer inject skipped (import: {e!r}).")
        return
    index_path = _Path(pipecat_ai_prebuilt.__file__).parent / "client" / "dist" / "index.html"
    if not index_path.is_file():
        logger.warning(f"Client jitter-buffer inject skipped (no index.html at {index_path}).")
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

    @app.middleware("http")
    async def _inject_jitter_buffer(request, call_next):
        # Only the index page (exact /client or /client/); assets pass through to the mount.
        if request.method == "GET" and request.url.path in ("/client", "/client/"):
            try:
                html = index_path.read_text(encoding="utf-8").replace("<head>", "<head>" + patch, 1)
                return HTMLResponse(html)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Jitter-buffer inject failed; serving default page: {e!r}")
        return await call_next(request)

    logger.info(f"Client jitter buffer ENABLED: {ms}ms (CLIENT_JITTER_BUFFER_MS=0 to disable).")


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
    # Pin ICE host candidates to the Tailscale interface so the stable 100.x<->100.x pair
    # wins immediately (kills the intermittent-mic ICE pollution -- see the function docstring).
    _restrict_ice_to_subnet()
    # Add STUN so the server discovers its public IP on GCP / any 1:1-NAT host.
    _configure_stun_servers()
    from pipecat.runner.run import main

    main()
