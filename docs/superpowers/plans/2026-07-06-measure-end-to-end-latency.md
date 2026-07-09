# Measure End-to-End Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the measure system report the actual per-stage latency all the way to the browser output the user hears, as a waterfall that sums to a true end-to-end number.

**Architecture:** Stitch the same-box headless probe's `time.time()` arrivals onto the pipeline log's `t0` (`t0.timestamp()`), detect the answer-audio onset at the client, and build a per-stage waterfall. Add an opt-in real-browser playout-onset beacon as a second, truer last-mile source. All measurement lives in `scripts/measure.py`; the only pipeline change is one env-gated, default-OFF client `<head>` injector + a beacon endpoint in `pipeline/main.py`. `TtfoMeter` is untouched.

**Tech Stack:** Python 3.11 (system env — has pipecat/aiortc/numpy/av), pytest 9.1.1, vanilla JS `<head>` injection into the pipecat prebuilt bundle, static HTML/JS report.

## Global Constraints

- **Server `.py` source stays ASCII-safe** (Windows console is cp1252; use `--` and `->`, never `—`/`→` in `.py`). Markdown/HTML may use full Unicode.
- **The prebuilt `/client` bundle is never forked.** New client behavior is an env-gated patch registered into `_client_head_patches` via `_ensure_client_patch_middleware()` — the ONE sanctioned mechanism.
- **New client instrumentation defaults OFF** (measurement scaffolding), same convention as `CLIENT_AV_STATS_MONITOR`.
- **`pipeline/metrics.py` / `TtfoMeter` must not change** — the live acceptance metric stays as-is.
- Test files follow the repo convention: `archive/_<name>_test.py`, runnable as `python -m archive._<name>_test` AND collectable by pytest.
- Run `python -m scripts.preflight` after touching `pipeline/main.py` (fragile pipecat imports).

---

### Task 1: Pure latency functions in measure.py (clock-stitch, onset, waterfall)

The testable core: three pure functions with zero network/GPU. Everything else builds on these.

**Files:**
- Modify: `scripts/measure.py` (add three functions + `import os`)
- Test: `archive/_measure_waterfall_test.py` (create)

**Interfaces:**
- Produces:
  - `answer_onset_epoch(samples, t0_epoch, guard=0.15, thresh_frac=0.18, run=3) -> float | None`
    where `samples` is `list[tuple[float, float]]` of `(arrival_epoch, rms)`.
  - `build_waterfall(anchors, playout_source="est") -> list[dict]` where `anchors` is a dict of
    t0-relative offsets (seconds) with keys `llm_recv, llm_ttfb, tts_recv, tts_ttfb, bot_started,
    client_arrival, playout` (any may be `None`). Each row dict: `{stage, delta, cum, source, status}`.
  - `parse_playout_beacon(lines, t0_dt) -> float | None` where `lines` is `list[tuple[datetime, str]]`.

- [ ] **Step 1: Write the failing tests**

Create `archive/_measure_waterfall_test.py`:

```python
"""Pure latency functions for the measure waterfall — onset, waterfall sum, beacon parse.
Run: python -m archive._measure_waterfall_test  (or: pytest archive/_measure_waterfall_test.py)"""
from datetime import datetime

from scripts.measure import answer_onset_epoch, build_waterfall, parse_playout_beacon


def test_onset_ignores_greeting_and_silence():
    # greeting burst BEFORE t0, then silence, then the real answer after t0.
    t0 = 1000.0
    samples = (
        [(998.0 + i * 0.02, 0.5) for i in range(10)]   # greeting, pre-t0 -> ignored
        + [(1000.0 + i * 0.02, 0.0) for i in range(20)]  # silence after t0 -> below thresh
        + [(1000.4 + i * 0.02, 0.6) for i in range(20)]  # answer -> onset here
    )
    onset = answer_onset_epoch(samples, t0)
    assert onset is not None
    assert abs(onset - 1000.4) < 1e-6


def test_onset_needs_a_sustained_run_not_a_single_spike():
    t0 = 0.0
    samples = [(0.2, 0.9)] + [(0.2 + i * 0.02, 0.0) for i in range(1, 10)] \
        + [(0.6 + i * 0.02, 0.9) for i in range(5)]
    onset = answer_onset_epoch(samples, t0, run=3)
    assert abs(onset - 0.6) < 1e-6  # the lone 0.2s spike is skipped


def test_onset_all_silence_returns_none():
    assert answer_onset_epoch([(1.0, 0.0), (1.1, 0.0), (1.2, 0.0)], 0.0) is None


def test_waterfall_deltas_telescope_to_total():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=2.45,
                   bot_started=2.75, client_arrival=2.97, playout=3.12)
    rows = build_waterfall(anchors, playout_source="browser")
    ok = [r for r in rows if r["status"] == "ok"]
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(sum(r["delta"] for r in ok) - total) < 1e-6
    assert abs(total - 3.12) < 1e-6
    assert ok[-1]["source"] == "browser"


def test_waterfall_missing_anchor_is_unknown_and_sum_still_holds():
    anchors = dict(llm_recv=0.0, llm_ttfb=0.68, tts_recv=1.05, tts_ttfb=None,
                   bot_started=2.75, client_arrival=2.97, playout=None)
    rows = build_waterfall(anchors)
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["TTS synth first chunk"]["status"] == "unknown"
    ok = [r for r in rows if r["status"] == "ok"]
    total = [r for r in rows if r["status"] == "total"][0]["cum"]
    assert abs(sum(r["delta"] for r in ok) - total) < 1e-6  # telescoping survives the gap
    assert abs(total - 2.97) < 1e-6  # last known cum (playout missing -> est. fills in caller)


def test_parse_playout_beacon_offset():
    t0 = datetime.fromtimestamp(1751800000.0)
    lines = [
        (datetime.fromtimestamp(1751799999.0), "something before t0 audio-onset t=1"),
        (datetime.fromtimestamp(1751800000.5), '[client-playout] {"ev":"audio-onset","t":1751800000123}'),
    ]
    assert parse_playout_beacon(lines, t0) == 0.123


def main():
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("PASS _measure_waterfall_test")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest archive/_measure_waterfall_test.py -q`
Expected: FAIL — `ImportError: cannot import name 'answer_onset_epoch' from 'scripts.measure'`.

- [ ] **Step 3: Add the three functions to `scripts/measure.py`**

Add `import os` to the import block (after `import json`). Then add these functions after
`parse_turn` (before `build_events`):

```python
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
    ("STT finalize -> LLM", "llm_recv", "log"),
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
    Returns ordered rows: {stage, delta, cum, source, status}; the final 'total' row carries the
    end-to-end cum. The last stage's source is `playout_source` (browser | est).
    """
    rows, prev = [], 0.0
    for label, key, source in _WATERFALL_STAGES:
        if key == "playout":
            source = playout_source
        end = anchors.get(key)
        if end is None:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest archive/_measure_waterfall_test.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/measure.py archive/_measure_waterfall_test.py
git commit -m "feat(measure): pure clock-stitch, answer-onset, and waterfall builder (TDD)"
```

---

### Task 2: Wire the headless source into the report (audio-pump RMS + waterfall)

Feed real data into Task 1's functions: capture per-frame audio RMS in the probe, compute the
client answer-arrival, build the waterfall into the report and the printed summary.

**Files:**
- Modify: `scripts/measure.py` (`run_probe`, `probe_metrics`, `main`, `print_summary`, argparse)

**Interfaces:**
- Consumes: `answer_onset_epoch`, `build_waterfall` (Task 1).
- Produces: `report["waterfall"]` (list of rows) in `output/measure_report.json` +
  `docs/measure_data.js`; a `--from-browser` flag on the CLI.

- [ ] **Step 1: Add an RMS helper + split the probe pumps**

In `scripts/measure.py`, add this helper just above `run_probe`:

```python
def _audio_rms(frame):
    """RMS of one decoded aiortc AudioFrame (int16 PCM) -> float."""
    s = frame.to_ndarray()
    if s.size == 0:
        return 0.0
    s = s.astype(np.float64)
    return float(np.sqrt(np.mean(s * s)))
```

In `run_probe`, replace the single `pump` coroutine and its two `ensure_future` wirings. Delete:

```python
    async def pump(track, sink):
        while True:
            try:
                await track.recv()
            except Exception:
                return
            sink.append(time.time())
```

and replace the recorder-wiring block with:

```python
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
```

(Now `awall` is a list of `(epoch, rms)` tuples; `vwall` stays a list of floats.)

- [ ] **Step 2: Fix the one `awall` consumer in `probe_metrics`**

In `probe_metrics`, the audio-gap block reads `awall` as bare floats. Change:

```python
    if len(awall) > 2:
        ag = np.diff(np.array(awall))
```

to:

```python
    if len(awall) > 2:
        ag = np.diff(np.array([t for t, _ in awall]))
```

- [ ] **Step 3: Build the waterfall in `main` and support `--from-browser`**

Replace the body of `async def main(args)` with:

```python
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
    turn = parse_turn()
    t0_epoch = turn["t0"].timestamp()

    # Last mile source 1 (headless): first ANSWER audio arriving at the client, on the log clock.
    client_arrival = None
    if awall:
        onset = answer_onset_epoch(awall, t0_epoch)
        client_arrival = round(onset - t0_epoch, 3) if onset else None

    # Last mile source 2 (real browser): the [client-playout] onset beacon, if present.
    playout = None
    if args.from_browser:
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
        playout_source = "browser"
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
```

- [ ] **Step 4: Print the waterfall in `print_summary`**

In `print_summary`, add this block just before the `print(f"wrote {JSON_OUT}")` line:

```python
    print("latency waterfall (t0 = user stopped -> user hears):")
    for r in report.get("waterfall", []):
        d = f"{r['delta']:+.2f}s" if r["delta"] is not None else "   ?  "
        c = f"{r['cum']:.2f}s" if r["cum"] is not None else "  ?  "
        src = f"[{r['source']}]" if r["source"] else ""
        print(f"  {r['stage']:<36} {d:>8}   cum {c:>7}  {src}")
```

- [ ] **Step 5: Add the `--from-browser` flag**

In the argparse block, after the `--offline-capture` argument, add:

```python
    ap.add_argument("--from-browser", action="store_true",
                    help="parse-only: use a real browser's [client-playout] beacon for the last "
                         "mile instead of driving the headless probe (open /client, do one turn, "
                         "with CLIENT_PLAYOUT_PROBE=1)")
```

- [ ] **Step 6: Re-run Task 1 tests (no regression) + live verify**

Run: `python -m pytest archive/_measure_waterfall_test.py -q`
Expected: PASS (6 passed).

Then, with the full stack up (CosyVoice `:8001`, MuseTalk `:8002`, pipeline `:7860`; no `/client` tab open):
Run: `python -m scripts.measure --mic output/q_ai.wav`
Expected: a `latency waterfall` prints; the `Transport + encode + network` delta is positive and
small (roughly tens–hundreds of ms); `END-TO-END, user hears` cum ~= `[TTFO]` + that last mile.
(If `output/q_ai.wav` is absent, use any question wav, e.g. `output/_zh_q.wav`.)

- [ ] **Step 7: Commit**

```bash
git add scripts/measure.py
git commit -m "feat(measure): headless client-arrival + waterfall in report and summary"
```

---

### Task 3: Real-browser playout-onset beacon (main.py injector + endpoint)

The opt-in truth layer: a default-OFF client injector that beacons the instant the voice actually
starts playing, and the endpoint that logs it. `parse_playout_beacon` (Task 1) already consumes it.

**Files:**
- Modify: `pipeline/main.py` (add `/client/playout` branch in `_inject_client_patches`; add
  `_install_client_playout_probe()`; call it in `__main__`)

**Interfaces:**
- Consumes: `_ensure_client_patch_middleware()`, `_client_head_patches`, `ensure_file_sink`
  (all existing in `main.py`).
- Produces: `[client-playout] {"ev":"audio-onset","t":<epoch_ms>}` lines in `pipeline.log`,
  gated by `CLIENT_PLAYOUT_PROBE=1`.

- [ ] **Step 1: Add the beacon endpoint branch**

In `_inject_client_patches` (inside `_ensure_client_patch_middleware`), directly after the
`/client/av-stats` branch's `return HTMLResponse("", status_code=204)` and before the
`# Only the index page` comment, add:

```python
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
```

- [ ] **Step 2: Add the injector function**

Add `_install_client_playout_probe()` just after `_install_client_av_stats_monitor()` (before
`_restrict_ice_to_subnet`):

```python
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
```

- [ ] **Step 3: Wire it into `__main__`**

In `pipeline/main.py`'s `__main__`, immediately after the `_install_client_av_stats_monitor()`
call, add:

```python
    # Real-browser voice-onset beacon (to-the-ear last mile for the measure waterfall).
    _install_client_playout_probe()
```

- [ ] **Step 4: Preflight + import check**

Run: `python -m scripts.preflight`
Expected: exits 0 (all fragile imports resolve).

Run: `python -c "import ast; ast.parse(open('pipeline/main.py',encoding='utf-8').read()); print('main.py parses')"`
Expected: `main.py parses`.

- [ ] **Step 5: Live verify the beacon + parse path**

Restart the pipeline with the probe armed (from the repo root):
Run: `CLIENT_PLAYOUT_PROBE=1 python -m pipeline.main`  (or set `CLIENT_PLAYOUT_PROBE=1` in `.env`)
Open `http://localhost:7860/client/` (WITH trailing slash), do one spoken turn, then:
Run: `grep -c "client-playout" logs/pipeline.log`
Expected: >= 1 (a `[client-playout] {"ev":"audio-onset","t":...}` line was logged).

Then close the tab and:
Run: `python -m scripts.measure --from-browser`
Expected: the waterfall's `Browser jitter + decode + playout` row shows source `[browser]` (not
`[est]`) and a filled cum.

- [ ] **Step 6: Commit**

```bash
git add pipeline/main.py
git commit -m "feat(measure): opt-in real-browser voice-onset beacon (to-the-ear last mile)"
```

---

### Task 4: Render the waterfall in the timeline HTML

Surface the new `waterfall` array in `docs/workflow-timeline.html` (which auto-loads
`measure_data.js`). Reuses the existing `.grid3`/`.card` styling — no new CSS.

**Files:**
- Modify: `docs/workflow-timeline.html` (one HTML block + one JS function + one call)

**Interfaces:**
- Consumes: `window.MEASURE.waterfall` (Task 2 writes it).

- [ ] **Step 1: Add the panel markup**

In `docs/workflow-timeline.html`, directly after `<div class="events" id="events"></div>`
(the events section), insert:

```html

  <h2>Where the time goes — to the user's ear</h2>
  <div class="grid3" id="waterfall"></div>
  <p class="sub" style="margin-top:10px">Each card is one stage's cost; the stages add up from
  t0 (user stopped speaking) to the moment the browser plays the first sound.
  <b>log</b> = pipeline clock · <b>probe</b> = headless client arrival ·
  <b>browser</b>/<b>est.</b> = real speaker playout or a jitter-buffer estimate. A
  <span style="color:var(--warn)">warn</span> card = that stage's anchor was missing this run.</p>
```

- [ ] **Step 2: Add the renderer and its default**

In the `<script>`, after the line `const METRICS  = (_M && _M.metrics)  || METRICS_DEFAULT;`
(~line 336), add:

```javascript
const WATERFALL = (_M && _M.waterfall) || [];
```

Then add this function next to `buildMetrics` (after it, ~line 457):

```javascript
function buildWaterfall(){
  const el=document.getElementById('waterfall');
  if(!el) return;
  if(!WATERFALL.length){ el.innerHTML='<div class="card"><div class="n">Run '+
    '<code>python -m scripts.measure</code> to populate.</div></div>'; return; }
  WATERFALL.forEach(r=>{
    const card=document.createElement('div'); card.className='card';
    const tag = r.status==='total' ? ' tag-ok' : (r.status==='unknown' ? ' tag-warn' : '');
    const dv  = r.delta!=null ? (r.status==='total' ? '' : '+')+r.delta.toFixed(2)+'s'
              : (r.status==='total' ? '' : '?');
    const val = r.status==='total' && r.cum!=null ? r.cum.toFixed(2)+'s' : dv;
    const sub = r.status==='total' ? 'end-to-end, user hears'
              : (r.cum!=null ? 'cum '+r.cum.toFixed(2)+'s' : 'anchor missing')
                + (r.source ? ' · '+r.source : '');
    card.innerHTML=`<div class="k">${r.stage}</div>`+
      `<div class="v${tag}">${val}</div>`+
      `<div class="n">${sub}</div>`;
    el.appendChild(card);
  });
}
```

- [ ] **Step 3: Call it on load**

Find where `buildMetrics()` is called (near the other `build*()` calls at the bottom of the
script) and add `buildWaterfall();` on the next line. If `buildMetrics()` is not called via a
bare statement, add `buildWaterfall();` immediately after the `buildHandoffs();` call.

- [ ] **Step 4: Verify it renders**

Run: `python -c "import json,pathlib; d=json.loads(pathlib.Path('output/measure_report.json').read_text(encoding='utf-8')); print('waterfall rows:', len(d.get('waterfall',[])))"`
Expected: `waterfall rows: 8` (7 stages + total), assuming Task 2's live run wrote the report.

Open `docs/workflow-timeline.html` in a browser (hard-refresh). Expected: a "Where the time goes"
section shows one card per stage with `+Δs` and `cum`, ending in a green end-to-end card.

- [ ] **Step 5: Commit**

```bash
git add docs/workflow-timeline.html
git commit -m "feat(measure): render the latency-to-the-ear waterfall in the timeline HTML"
```

---

## Self-Review notes

- **Spec coverage:** clock-stitch → Task 1 (`answer_onset_epoch` uses the log-clock t0 epoch) +
  Task 2 (`t0.timestamp()`); headless source → Task 2; browser source → Task 1
  (`parse_playout_beacon`) + Task 3 (injector/endpoint) + Task 2 (`--from-browser`); waterfall
  deliverable → Task 1 (`build_waterfall`) + Task 4 (render); graceful degradation → Task 1
  `unknown` rows + Task 2 estimate fallback; `metrics.py` untouched (no task touches it).
- **Estimate vs measured labeling:** Task 2 sets `playout_source` to `browser`/`est`; Task 4
  shows it in the card subtitle.
- **Types:** `awall` is `list[(epoch,rms)]` from Task 2 on; Task 1's `answer_onset_epoch` and the
  Task 2 `apump`/`probe_metrics` consumer all agree on that shape. `anchors` keys match between
  `build_waterfall` (`_WATERFALL_STAGES`) and Task 2's `anchors` dict.
- **Beacon format:** injector emits JSON `{"ev":"audio-onset","t":<ms>}`; endpoint logs it under
  `[client-playout]`; `parse_playout_beacon` matches `"t":\s*<digits>` — consistent across Tasks 1/3.
