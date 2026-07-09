"""Preflight / smoke test — run before running the pipeline.

Pipecat's import paths drift between releases, and the stage factories import
provider SDKs lazily. This resolves every fragile import *without needing API
keys or network*, so you catch version drift in one shot.

For each stage it reports one of:
  PASS  imports resolve and the service constructs
  KEYS  imports resolve; construction needs keys/config (expected, fine)
  DEPR  constructs, but on a deprecated Pipecat API — soft drift; fix soon
  DRIFT import failed — a Pipecat path moved or an extra isn't installed (FIX)
  SKIP  optional local dep not installed yet (the avatar's torch/onnx stack,
        which only lives in the `musetalk` conda env)

Exit code is non-zero if any DRIFT is found, so it doubles as a CI gate. DEPR is
non-fatal (it's the early-warning net for the *next* hard drift), so it does not
change the exit code.

    python -m scripts.preflight
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Repo root, so we can tell *our* deprecated call sites (actionable drift) from
# transitive stdlib/third-party ones (audioop, importlib_resources, ...) that
# we can't fix and don't want crying wolf on every run.
ROOT = Path(__file__).resolve().parent.parent

# Local avatar deps that are expected to be absent outside the `musetalk` env. An
# import error mentioning one of these is a SKIP, not a Pipecat-drift failure.
OPTIONAL_DEPS = {"websockets", "torch", "numpy", "cv2", "onnxruntime", "mediapipe", "filetype"}

RESET, GREEN, YELLOW, RED, GREY, ORANGE = (
    "\033[0m", "\033[32m", "\033[33m", "\033[31m", "\033[90m", "\033[38;5;208m"
)


def _check(label: str, fn) -> str:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        deps = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and str(ROOT) in str(Path(w.filename).resolve())
        ]
        if deps:
            _check.depr = True
            # de-dupe while preserving order, then show concisely
            seen = list(dict.fromkeys(deps))
            return f"{ORANGE}DEPR{RESET}  {label}  -> {'; '.join(seen)}"
        return f"{GREEN}PASS{RESET}  {label}"
    except (ImportError, ModuleNotFoundError) as e:
        missing = getattr(e, "name", "") or str(e)
        if any(dep in str(e) for dep in OPTIONAL_DEPS):
            return f"{GREY}SKIP{RESET}  {label}  ({missing} not installed)"
        _check.drift = True
        return f"{RED}DRIFT{RESET} {label}  -> {e}"
    except Exception as e:  # noqa: BLE001 — construction reached, just needs config
        return f"{YELLOW}KEYS{RESET}  {label}  ({type(e).__name__})"


_check.drift = False
_check.depr = False


def main() -> int:
    print("== Environment ==")
    try:
        import pipecat

        print(f"  pipecat {getattr(pipecat, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {RED}pipecat not importable: {e}{RESET}")
        print("  -> pip install -r requirements.txt")
        return 2

    from pipeline.config import config
    from pipeline.stages import build_avatar, build_llm, build_stt, build_tts, build_vad_params

    print("\n== Core modules ==")
    print(_check("pipeline.metrics", lambda: __import__("pipeline.metrics", fromlist=["TtfoMeter"])))
    print(_check("pipeline.main", lambda: __import__("pipeline.main", fromlist=["run_bot"])))
    print(_check("Silero VAD construct", build_vad_params))

    print("\n== Stages (single stack) ==")
    print(_check(f"stt  ({config.stt_provider})", lambda: build_stt(config)))
    # Cover the offline-STT import paths even when Deepgram is the active provider, so a
    # Pipecat path move (Segmented/STTService, VAD frames, utils.time) is caught on the default stack.
    print(_check("stt  (funasr wrapper import)",
                 lambda: __import__("local_services.funasr_stt", fromlist=["FunasrSTTService"])))
    print(_check("stt  (sherpa wrapper import)",
                 lambda: __import__("local_services.sherpa_stt", fromlist=["SherpaStreamingSTTService"])))
    print(_check("llm  (OpenRouter)", lambda: build_llm(config)))
    print(_check(f"tts  ({config.tts_provider})", lambda: build_tts(config)))
    print(_check("avatar (musetalk)", lambda: build_avatar(config)))

    print("\n== Support ==")
    print(_check("log_setup", lambda: __import__("log_setup", fromlist=["setup_logging"])))

    print()
    if _check.drift:
        print(f"{RED}Drift detected — fix the imports above before running.{RESET}")
        return 1
    if _check.depr:
        # Non-fatal: still exit 0, but flag it so the next hard drift is caught early.
        print(f"{ORANGE}Deprecated API in use (DEPR above) — migrate before the next "
              f"Pipecat upgrade removes it.{RESET}")
        return 0
    print(f"{GREEN}No drift. Imports resolve for the active stack.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
