"""Config panel: a tiny web UI to view/edit .env and restart the pipeline, so the system
can be reconfigured from a browser (incl. remotely over Tailscale) without hand-editing the
file or touching the shell. Stdlib only; single-client; reads/writes the repo .env IN PLACE,
preserving comments + inline annotations (it only swaps the VALUE of known keys).

Run from the repo root with the SYSTEM python (the one that has pipecat, since Restart spawns
`python -m pipeline.main`):   python -m local_services.config_panel.server   -> http://localhost:7870
ASCII-only source.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 7870
REPO = Path(__file__).resolve().parents[2]   # .../visualllm
ENV = REPO / ".env"
_HTML = Path(__file__).parent / "index.html"
_PIPELINE_LOG = REPO / "scratchpad_pipeline.log"

# CUDA graphs (COSYVOICE_VLLM_EAGER) live in the CosyVoice repo's WSL launch script, NOT .env, and
# the pipeline Restart does NOT touch the WSL vLLM server. The panel toggles graphs by rewriting
# that script's default and relaunching the WSL server. Graphs ON == EAGER default 0 (2026-07-05
# re-investigation: graphs are faster + lower-variance in TTS TTFB; docs P32). Windows + WSL paths:
_COSY_SCRIPT = Path(r"E:\Claude\cosyvoice-local-tts\run_vllm_server.sh")
_COSY_SCRIPT_WSL = "/mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"
_WSL_DISTRO = "Ubuntu"
_EAGER_RE = re.compile(r"(COSYVOICE_VLLM_EAGER=\$\{COSYVOICE_VLLM_EAGER:-)([01])(\})")

# Fields the panel exposes. group: curated | advanced. type: select -> options; text -> free.
# Each maps 1:1 to a .env KEY. Unknown keys in .env are left untouched.
FIELDS = [
    # --- curated (the ones changed most) ---
    {"key": "LANGUAGE", "group": "curated", "label": "Language", "type": "select",
     "options": ["zh", "en", "th"], "help": "STT/LLM/voice language"},
    {"key": "LLM_PROVIDER", "group": "curated", "label": "LLM provider", "type": "select",
     "options": ["openrouter", "weather_chain"], "help": "openrouter = general chat (local Ollama or cloud); weather_chain = NCU weather bot"},
    {"key": "STT_PROVIDER", "group": "curated", "label": "STT provider", "type": "select",
     "options": ["deepgram", "sherpa", "funasr"], "help": "deepgram = cloud; sherpa = local offline streaming (~0 VRAM); funasr = local SenseVoice (:8004)"},
    {"key": "OPENROUTER_MODEL", "group": "curated", "label": "Chat model", "type": "text",
     "options": ["meta-llama/llama-4-scout", "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.5-flash-lite", "qwen2.5:7b"],
     "help": "Used when LLM provider = openrouter. Baseline = llama-4-scout (Groq, fast, clean zh)."},
    {"key": "OPENROUTER_PROVIDER_ONLY", "group": "curated", "label": "Pin LLM backend", "type": "text",
     "options": ["Groq", ""], "help": "Pin OpenRouter to one fast backend (Groq) -- kills the LLM-hop tail. Empty = unpinned."},
    {"key": "WEATHER_CHAIN_MODEL", "group": "curated", "label": "Weather model", "type": "text",
     "options": ["qwen2.5:7b", "gemma3:27b"], "help": "Used when LLM provider = weather_chain (must be installed on NCU)"},
    {"key": "TTS_PROVIDER", "group": "curated", "label": "TTS provider", "type": "select",
     "options": ["cosyvoice", "moss", "elevenlabs", "deepgram"], "help": "Voice engine"},
    {"key": "COSYVOICE_VOICE", "group": "curated", "label": "CosyVoice voice", "type": "text",
     "options": ["weather", "pro"], "help": "Registered zero-shot speaker id (CosyVoice)"},
    {"key": "MUSETALK_SYNC_MODE", "group": "curated", "label": "A/V sync", "type": "select",
     "options": ["steady", "live"], "help": "steady = synced start (voice may pause under load); live = voice instant, lips trail"},
    {"key": "FILLER_WORDS", "group": "curated", "label": "Filler opener", "type": "select",
     "options": ["1", "0"], "help": "1 = open the turn on a 'thinking' phrase so the avatar starts ~0.7s sooner (perception win). zh may feel delayed (P30)."},
    {"key": "COSYVOICE_FIRST_PIECE", "group": "curated", "label": "First-clause split", "type": "select",
     "options": ["1", "0"], "help": "1 = speak the opening clause early -> lower TTS first-chunk (TTFO win). Needed by filler opener."},
    {"key": "AVATAR_MEMORY", "group": "curated", "label": "Avatar memory", "type": "select",
     "options": ["1", "0"], "help": "1 = remember across turns (local CPU qwen); 0 = stateless"},

    # --- advanced ---
    {"key": "TTFO_TARGET_SECONDS", "group": "advanced", "label": "TTFO target (s)", "type": "text"},
    {"key": "ECHO_GUARD", "group": "advanced", "label": "Echo guard", "type": "select", "options": ["0", "1"],
     "help": "0 = barge-in (headphones); 1 = half-duplex mute (only valid with sync=live)"},
    {"key": "OPENROUTER_BASE_URL", "group": "advanced", "label": "Chat base URL", "type": "text",
     "options": ["http://localhost:11434/v1", "https://openrouter.ai/api/v1"]},
    {"key": "MEMORY_LLM_MODEL", "group": "advanced", "label": "Memory model", "type": "text"},
    {"key": "MEMORY_LLM_GATED", "group": "advanced", "label": "Memory gated", "type": "select", "options": ["1", "0"]},
    {"key": "COSYVOICE_URL", "group": "advanced", "label": "CosyVoice URL", "type": "text"},
    {"key": "MOSS_URL", "group": "advanced", "label": "MOSS URL", "type": "text"},
    {"key": "AVATAR_REF", "group": "advanced", "label": "Avatar portrait", "type": "text"},
    {"key": "MUSETALK_SIZE", "group": "advanced", "label": "Frame px", "type": "text", "options": ["256", "512"]},
    {"key": "MUSETALK_FPS", "group": "advanced", "label": "FPS", "type": "text", "options": ["8", "10", "12", "14", "16", "20"]},
    {"key": "MUSETALK_LEAD_FRAMES", "group": "advanced", "label": "Lead frames", "type": "text"},
    {"key": "MUSETALK_IDLE_MOTION", "group": "advanced", "label": "Idle motion", "type": "select", "options": ["0", "1"]},
    {"key": "MUSETALK_CLOSE_FADE_FRAMES", "group": "advanced", "label": "Close fade frames", "type": "text"},
    {"key": "FILLER_WORDS_COUNT", "group": "advanced", "label": "Filler count", "type": "text",
     "help": "How many thinking-phrases to chain when Filler opener = 1 (a longer opener)"},
    {"key": "COSYVOICE_FIRST_PIECE_ZH", "group": "advanced", "label": "First-clause split (zh)", "type": "select",
     "options": ["1", "0"], "help": "1 = split the zh opener at a full-width comma (the en char-split never fires on zh)"},
    {"key": "CLIENT_FORCE_SPEAKER", "group": "advanced", "label": "Force phone speaker", "type": "select",
     "options": ["1", "0"], "help": "1 = play voice on the phone loudspeaker (mobile UA only); desktop untouched"},
    {"key": "CLIENT_JITTER_BUFFER_MS", "group": "advanced", "label": "Jitter buffer (ms)", "type": "text"},
    {"key": "WEBRTC_VIDEO_BITRATE_MAX", "group": "advanced", "label": "Max video bitrate", "type": "text"},
    {"key": "WEBRTC_ICE_SUBNET", "group": "advanced", "label": "ICE subnet", "type": "text"},
]
_KNOWN = {f["key"] for f in FIELDS}

# Servers to show health for: label -> port.
PORTS = {"pipeline": 7860, "avatar": 8002, "cosyvoice": 8001, "moss": 8003,
         "memory-sim": 7900, "ollama": 11434}


# --------------------------------------------------------------------------- env IO
def read_env() -> dict:
    """key -> current value (inline `# comment` stripped), for known keys present in .env."""
    vals = {}
    if not ENV.exists():
        return vals
    for line in ENV.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=(.*)$", line)
        if not m:
            continue
        key, rest = m.group(1), m.group(2)
        val = rest.split("#", 1)[0].strip()   # none of our values contain '#'
        vals[key] = val
    return vals


def write_env(updates: dict) -> list:
    """Swap the VALUE of each known key in place, preserving the rest of the line
    (alignment + inline comment). Returns the list of keys actually changed."""
    lines = ENV.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = []
    seen = set()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Z0-9_]+)(\s*=)(.*?)(\r?\n)?$", line)
        if not m:
            continue
        indent, key, eq, rest, nl = m.groups()
        nl = nl or "\n"
        if key not in updates:
            continue
        seen.add(key)
        new_val = str(updates[key])
        # keep an inline comment if there was one
        if "#" in rest:
            cur_val, comment = rest.split("#", 1)
            pad = cur_val[len(cur_val.rstrip()):] or "   "
            new_rest = f"{new_val}{pad}#{comment}"
        else:
            new_rest = new_val
        old_val = rest.split("#", 1)[0].strip()
        if old_val != new_val:
            lines[i] = f"{indent}{key}{eq}{new_rest}{nl}"
            changed.append(key)
    # append any updated keys that weren't already in the file
    for key, val in updates.items():
        if key not in seen:
            lines.append(f"{key}={val}\n")
            changed.append(key)
    ENV.write_text("".join(lines), encoding="utf-8")
    return changed


# --------------------------------------------------------------------------- status / restart
def port_up(port: int) -> bool:
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=8).stdout
    except Exception:  # noqa: BLE001
        return False
    return any((f":{port} " in ln and "LISTENING" in ln) for ln in out.splitlines())


def status() -> dict:
    return {name: port_up(p) for name, p in PORTS.items()}


def _pids_on(port: int) -> list:
    """PIDs LISTENING on a port, via netstat (PowerShell cmdlets are slow to spawn here)."""
    pids = set()
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=8).stdout
    except Exception:  # noqa: BLE001
        return []
    for ln in out.splitlines():
        if f":{port} " in ln and "LISTENING" in ln:
            parts = ln.split()
            if parts and parts[-1].isdigit():
                pids.add(parts[-1])
    return list(pids)


def restart_pipeline() -> dict:
    """Kill whatever listens on :7860, then start a fresh detached `python -m pipeline.main`.
    Always returns a JSON-able dict (never raises) so the panel shows a clean message."""
    try:
        for pid in _pids_on(7860):
            # Native TerminateProcess, not `taskkill`/PowerShell: those can hang for tens of
            # seconds on this box under CPU load (the bug that made Restart error out); the
            # Win32 call returns instantly.
            try:
                if os.name == "nt":
                    PROCESS_TERMINATE = 0x0001
                    h = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
                    if h:
                        ctypes.windll.kernel32.TerminateProcess(h, 1)
                        ctypes.windll.kernel32.CloseHandle(h)
                else:
                    os.kill(int(pid), 9)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2)
        DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        log = open(_PIPELINE_LOG, "ab")
        subprocess.Popen(
            [sys.executable, "-m", "pipeline.main"], cwd=str(REPO),
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=DETACHED if os.name == "nt" else 0,
        )
        for _ in range(40):  # wait up to ~40s for it to bind :7860
            time.sleep(1)
            if port_up(7860):
                return {"ok": True, "message": "pipeline restarted (bound :7860)"}
        return {"ok": False, "message": "started, but :7860 not up in 40s -- check scratchpad_pipeline.log"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"restart failed: {type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- CUDA graphs (WSL TTS)
def read_graphs():
    """True if CUDA graphs are ON (EAGER default 0) in the WSL launch script; None if unknown."""
    try:
        m = _EAGER_RE.search(_COSY_SCRIPT.read_bytes().decode("utf-8"))
        return (m.group(2) == "0") if m else None
    except Exception:  # noqa: BLE001
        return None


def write_graphs(on: bool) -> bool:
    """Set the EAGER default in the launch script (graphs on => EAGER 0). write_bytes, not
    write_text, so Windows does NOT rewrite the script's LF endings to CRLF. Returns True if changed."""
    try:
        raw = _COSY_SCRIPT.read_bytes().decode("utf-8")
    except Exception:  # noqa: BLE001
        return False
    new = _EAGER_RE.sub(lambda m: m.group(1) + ("0" if on else "1") + m.group(3), raw, count=1)
    if new != raw:
        _COSY_SCRIPT.write_bytes(new.encode("utf-8"))
    return new != raw


def _cosy_health_ok() -> bool:
    """HTTP /health at COSYVOICE_URL (the WSL IP -- Windows netstat can't see the WSL port)."""
    url = read_env().get("COSYVOICE_URL", "").strip()
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def restart_cosyvoice() -> dict:
    """Kill + relaunch the WSL vLLM CosyVoice server so a graphs change takes effect. Mirrors
    launch.ps1: a DETACHED wsl.exe running the script in the FOREGROUND (an `&`-backgrounded WSL
    child dies when its launching shell returns). Always returns a JSON-able dict."""
    try:
        subprocess.run(["wsl.exe", "-d", _WSL_DISTRO, "-e", "bash", "-c", "pkill -f 'uvicorn app:app'"],
                       timeout=20)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(2)
    try:
        (REPO / "logs").mkdir(exist_ok=True)
        log = open(REPO / "logs" / "cosyvoice_wsl.log", "ab")
        DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            ["wsl.exe", "-d", _WSL_DISTRO, "-e", "bash", "-c", "bash " + _COSY_SCRIPT_WSL],
            stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=DETACHED if os.name == "nt" else 0,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"relaunch failed: {type(e).__name__}: {e}"}
    for _ in range(75):  # graphs capture + model load: give ~150s
        time.sleep(2)
        if _cosy_health_ok():
            return {"ok": True, "message": "CosyVoice restarted -- graphs setting applied"}
    return {"ok": False, "message": "relaunched, but :8001 not healthy in ~150s -- check "
            "logs/cosyvoice_wsl.log (VRAM/load-order: start CosyVoice before MuseTalk, P15)"}


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if not _HTML.exists():
                return self._send(500, {"error": "index.html missing"})
            return self._send(200, _HTML.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if self.path == "/config":
            return self._send(200, {"fields": FIELDS, "values": read_env(), "graphs": read_graphs()})
        if self.path == "/status":
            return self._send(200, status())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/save":
            updates = {k: v for k, v in (self._body() or {}).items() if k in _KNOWN}
            if not updates:
                return self._send(400, {"error": "no known keys to save"})
            changed = write_env(updates)
            return self._send(200, {"ok": True, "changed": changed, "values": read_env()})
        if self.path == "/restart":
            return self._send(200, restart_pipeline())
        if self.path == "/graphs":
            on = bool((self._body() or {}).get("on"))
            write_graphs(on)
            res = restart_cosyvoice()
            res["graphs"] = read_graphs()
            return self._send(200, res)
        return self._send(404, {"error": "not found"})


def main():
    print("config panel -> http://localhost:%d   (.env at %s)" % (PORT, ENV))
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
