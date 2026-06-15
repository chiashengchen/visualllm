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

from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipeline.config import config
from pipeline.metrics import TtfoMeter
from pipeline.stages import build_avatar, build_llm, build_stt, build_tts, build_vad_params


async def run_bot(transport: BaseTransport) -> None:
    logger.info(
        f"Providers: stt={config.stt_provider} llm={config.llm_provider} "
        f"tts={config.tts_provider} avatar={config.avatar_provider} "
        f"lang={config.language}"
    )

    stt = build_stt(config)
    llm = build_llm(config)
    tts = build_tts(config)
    avatar = build_avatar(config)
    meter = TtfoMeter(target_s=config.ttfo_target_s)

    context = LLMContext([{"role": "system", "content": config.system_prompt}])
    # Mute the mic while the bot is speaking so it can't transcribe its own voice
    # (acoustic echo when running on a loudspeaker). Trade-off: this disables
    # barge-in -- the user can't interrupt the bot mid-sentence.
    aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),      # mic in (+ VAD set in transport params)
            stt,                    # speech -> text
            aggregator.user(),      # add user turn to context
            llm,                    # text -> streamed text
            tts,                    # text -> streamed audio
            avatar,                 # audio -> lip-synced video+audio
            meter,                  # measure TTFO
            transport.output(),     # -> browser
            aggregator.assistant(), # add bot turn to context
        ]
    )

    # VAD-based barge-in is available, but the AlwaysUserMuteStrategy above mutes
    # the mic while the bot speaks, so interruptions are suppressed during bot
    # speech (the echo-guard trade-off).
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    async def _warmup_llm():
        # Open the HTTPS connection to the LLM now, so the transpacific TLS
        # handshake is done before the user's first message (kills cold start).
        try:
            await llm._client.chat.completions.create(
                model=llm._settings.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            )
            logger.info("LLM connection pre-warmed.")
        except Exception as e:  # noqa: BLE001 — best-effort only
            logger.debug(f"LLM warmup skipped: {e}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(transport, client):
        logger.info("Client connected — warming LLM + sending greeting.")
        asyncio.create_task(_warmup_llm())   # warm the LLM in the background
        greeting = "嗨，我準備好了，請說。" if config.is_mandarin else "Hi, I'm ready — go ahead."
        # Speak a fixed greeting directly via TTS (no LLM round-trip needed).
        await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(transport, client):
        logger.info(f"Client disconnected. TTFO summary: {meter.summary()}")
        await task.cancel()

    await PipelineRunner().run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Entrypoint the Pipecat dev runner calls with a configured transport."""
    from pipecat.runner.utils import create_transport

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,       # required for the avatar video stream
            vad_analyzer=build_vad_params(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    import sys

    # Windows consoles default to cp1252; Pipecat's runner prints emoji. Force
    # UTF-8 so startup doesn't crash on the banner.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # Serve the prebuilt UI unchanged, but drive its in-conversation status line
    # to read "Waiting for avatar..." while the avatar warms up (before its video
    # actually appears — the real ready signal). No popup/overlay: we just retext
    # the prebuilt's own "Waiting for messages..." status node so the loading
    # state shows inside the conversation screen, then revert once the face is live.
    from pathlib import Path

    import pipecat_ai_prebuilt
    from fastapi.responses import HTMLResponse

    from pipecat.runner.run import app, main

    _dist = Path(pipecat_ai_prebuilt.__file__).resolve().parent / "client" / "dist"
    _html = (_dist / "index.html").read_text(encoding="utf-8")
    # Make the prebuilt's relative asset paths absolute under /client/.
    _html = _html.replace('href="./', 'href="/client/').replace('src="./', 'src="/client/')

    _inject = """
<style>
  @keyframes vllmpulse{0%,100%{opacity:1}50%{opacity:.5}}
  .vllm-loading{animation:vllmpulse 1.1s ease-in-out infinite}
</style>
<script>
(function(){
  var LOADING='Waiting for avatar...';
  var WAITING='Waiting for messages...';
  var done=false;

  function video(){return document.querySelector('video');}
  // The avatar face is genuinely up only once its video produces frames.
  function live(){var v=video();return !!v&&v.videoWidth>0&&v.readyState>=2&&!v.paused;}

  // The prebuilt's in-conversation status line (only exists once connected):
  //   <div class="text-muted-foreground text-sm">Waiting for messages...</div>
  function statusEl(){
    var els=document.querySelectorAll('div.text-muted-foreground.text-sm');
    for(var i=0;i<els.length;i++){var t=(els[i].textContent||'').trim();
      if(t===LOADING||t===WAITING)return els[i];}
    // fallback (in case the class changes): any leaf node with exact text.
    els=document.querySelectorAll('div,span,p');
    for(var j=0;j<els.length;j++){var n=els[j];
      if(n.children.length===0){var u=(n.textContent||'').trim();
        if(u===LOADING||u===WAITING)return n;}}
    return null;
  }

  setInterval(function(){
    var el=statusEl();
    if(!el){ done=false; return; }       // connect screen / disconnected -> reset
    if(live()){                          // avatar face is genuinely up
      if(!done){ done=true; el.classList.remove('vllm-loading');
        if((el.textContent||'').trim()===LOADING) el.textContent=WAITING; }
      return;
    }
    if(!done){                           // connected, avatar still warming up
      el.textContent=LOADING;
      el.classList.add('vllm-loading');
    }
  },300);
})();
</script>

<!-- Avatar source toggle: flips local MuseTalk <-> cloud Simli, applied on the
     next connection. Floating pill, phone-friendly. -->
<style>
  #vllm-avsw{position:fixed;top:env(safe-area-inset-top,10px);right:10px;z-index:99999;
    display:flex;align-items:center;gap:6px;background:rgba(20,20,24,.82);
    color:#fff;font:600 13px/1 system-ui,sans-serif;padding:7px 9px;border-radius:999px;
    box-shadow:0 2px 10px rgba(0,0,0,.35);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);}
  #vllm-avsw button{appearance:none;border:0;cursor:pointer;border-radius:999px;
    padding:6px 12px;font:inherit;color:#cfd2da;background:transparent;transition:.15s;}
  #vllm-avsw button.on{background:#3b82f6;color:#fff;}
  #vllm-avsw .lbl{opacity:.7;font-weight:500;padding-left:4px;}
  #vllm-avsw.busy{opacity:.6;pointer-events:none;}
</style>
<div id="vllm-avsw" title="Avatar source (applies on next connect)">
  <span class="lbl">Avatar</span>
  <button data-m="musetalk_local">Local</button>
  <button data-m="simli">Cloud</button>
</div>
<script>
(function(){
  var box=document.getElementById('vllm-avsw');
  var btns=box.querySelectorAll('button');
  function paint(mode){
    btns.forEach(function(b){ b.classList.toggle('on', b.dataset.m===mode); });
  }
  function load(){
    fetch('/avatar-mode').then(function(r){return r.json();})
      .then(function(j){ paint(j.mode); }).catch(function(){});
  }
  btns.forEach(function(b){
    b.addEventListener('click', function(){
      if(b.classList.contains('on')) return;
      box.classList.add('busy');
      fetch('/avatar-mode',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({mode:b.dataset.m})})
        .then(function(r){return r.json();})
        .then(function(j){
          paint(j.mode); box.classList.remove('busy');
          var cloud=j.mode==='simli';
          alert('Avatar set to '+(cloud?'CLOUD (Simli)':'LOCAL (MuseTalk)')+
                '.\\n\\nReconnect (leave & rejoin) to apply.'+
                (cloud?'':'\\nMake sure the local MuseTalk server is running.'));
        })
        .catch(function(){ box.classList.remove('busy'); alert('Could not change avatar mode.'); });
    });
  });
  load();
})();
</script>

<!-- Android earpiece-vs-speaker toggle. Two-way WebRTC + a mic captured with
     echo cancellation forces Android into "communication" audio mode, which
     routes the bot's voice to the EARPIECE. The only web-level lever that flips
     Android to the media/loudspeaker path is disabling echoCancellation/NS/AGC
     on the mic — but then the mic can hear the bot (possible echo). So we make
     it a toggle: default Earpiece (safe, no echo); tap Speaker to force the
     loudspeaker. Applied by patching getUserMedia, so it takes effect on the
     next connect. Android-only; iOS already uses the speaker. -->
<style>
  #vllm-spk{position:fixed;bottom:env(safe-area-inset-bottom,12px);right:10px;z-index:99999;
    display:flex;align-items:center;gap:6px;background:rgba(20,20,24,.82);color:#fff;
    font:600 13px/1 system-ui,sans-serif;padding:7px 9px;border-radius:999px;
    box-shadow:0 2px 10px rgba(0,0,0,.35);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);}
  #vllm-spk button{appearance:none;border:0;cursor:pointer;border-radius:999px;
    padding:6px 12px;font:inherit;color:#cfd2da;background:transparent;transition:.15s;}
  #vllm-spk button.on{background:#3b82f6;color:#fff;}
  #vllm-spk .lbl{opacity:.7;font-weight:500;padding-left:4px;}
</style>
<script>
(function(){
  if(!/Android/i.test(navigator.userAgent||'')) return;
  var KEY='vllm_speaker';
  function spkOn(){ return localStorage.getItem(KEY)==='1'; }

  // Patch getUserMedia so the chosen audio routing applies on the next connect.
  var md=navigator.mediaDevices;
  if(md && md.getUserMedia){
    var orig=md.getUserMedia.bind(md);
    md.getUserMedia=function(c){
      try{
        if(spkOn() && c && c.audio){
          var a=(c.audio===true)?{}:Object.assign({},c.audio);
          a.echoCancellation=false; a.noiseSuppression=false; a.autoGainControl=false;
          c=Object.assign({},c,{audio:a});
        }
      }catch(e){}
      return orig(c);
    };
  }

  var box=document.createElement('div'); box.id='vllm-spk';
  box.innerHTML='<span class="lbl">Sound</span>'+
    '<button data-s="0">Earpiece</button><button data-s="1">Speaker</button>';
  function paint(){ box.querySelectorAll('button').forEach(function(b){
    b.classList.toggle('on', (b.dataset.s==='1')===spkOn()); }); }
  box.querySelectorAll('button').forEach(function(b){
    b.addEventListener('click', function(){
      var want=b.dataset.s==='1';
      if(want===spkOn()){ return; }
      localStorage.setItem(KEY, want?'1':'0'); paint();
      alert('Sound set to '+(want?'SPEAKER (loudspeaker)':'EARPIECE')+'.\\n\\n'+
        'Reconnect (leave & rejoin) to apply.'+
        (want?'\\nNote: on speaker the mic may pick up the bot (slight echo).':''));
    });
  });
  function mount(){ if(!document.body){ return setTimeout(mount,200);} document.body.appendChild(box); paint(); }
  mount();
})();
</script>
"""
    _html = _html.replace("</body>", _inject + "</body>")

    async def _client_with_overlay():
        # no-store so the browser never serves a stale /client after a restart.
        return HTMLResponse(_html, headers={"Cache-Control": "no-store"})

    # Registered before the runner mounts the prebuilt assets -> these win for
    # the HTML, while /client/assets/* still come from the prebuilt mount.
    for _p in ("/client", "/client/"):
        app.add_api_route(_p, _client_with_overlay, include_in_schema=False)

    # --- avatar source toggle (in-UI Local/Cloud switch) ---------------------
    # Reads/writes the same avatar_mode.txt that build_avatar() consults on every
    # connection, so a switch applies on the next reconnect (no restart needed).
    from fastapi import Request
    from fastapi.responses import JSONResponse

    from pipeline.stages.avatar import _MODE_FILE

    _VALID_MODES = {"musetalk_local", "simli", "heygen"}

    def _current_mode() -> str:
        try:
            v = _MODE_FILE.read_text(encoding="utf-8").strip()
            if v:
                return v
        except Exception:  # noqa: BLE001
            pass
        return config.avatar_provider

    async def _get_avatar_mode():
        return JSONResponse({"mode": _current_mode()})

    async def _set_avatar_mode(request: Request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        mode = (body or {}).get("mode", "").strip()
        if mode not in _VALID_MODES:
            return JSONResponse(
                {"error": f"invalid mode; expected one of {sorted(_VALID_MODES)}"},
                status_code=400,
            )
        _MODE_FILE.write_text(mode, encoding="utf-8")
        logger.info(f"Avatar mode set via UI -> {mode} (applies on next connect)")
        return JSONResponse({"mode": mode})

    app.add_api_route("/avatar-mode", _get_avatar_mode, methods=["GET"],
                      include_in_schema=False)
    app.add_api_route("/avatar-mode", _set_avatar_mode, methods=["POST"],
                      include_in_schema=False)

    main()
