# What To Do Next — VisualLLm

Plain checklist for the next session. Full context is in **`STATUS.md`** (source of
truth) and **`WORKFLOW.md`** (end-to-end run + `.env` reference). The system works;
this is what's left.

---

## 0. Start the system (3 processes)
- `cd E:\Claude\VisualLLm`
- **TTS (CosyVoice on vLLM, in WSL):**
  `wsl -d Ubuntu -e bash -c "bash /mnt/e/Claude/cosyvoice-local-tts/run_vllm_server.sh"`
  — wait for `Uvicorn running on http://0.0.0.0:8001`. Then make sure `.env`
  `COSYVOICE_URL` is the **WSL IP** (`wsl hostname -I`), **NOT localhost**.
- **Avatar server + pipeline:** `.\scripts\run.ps1` (picks the engine from `AVATAR`).
- Open `http://localhost:7860/client/` (**trailing slash**), allow the mic, wait for
  the face, then talk. Remote: the Tailscale HTTPS URL in a native browser (never RDP).

## 1. Live-confirm this session's wins (the one open item)
Three fixes landed 2026-06-22 and need a real call to confirm the *felt* result:
- **Remote mic** should now hold across turns (ICE pinned to Tailscale). If it still
  drops, watch the fresh pipeline log for the selected ICE pair.
- **CosyVoice on vLLM** should make her start replying ~2s sooner AND tighten the lips
  (frees the shared GPU). If TTS errors after a WSL reboot, the WSL IP changed —
  `wsl hostname -I` and update `COSYVOICE_URL` (or set up WSL mirrored networking).

## 1b. Avatar-timing fixes landed 2026-06-22 evening (verified headless; confirm on a call)
- **`cudnn.benchmark=False`** killed the ~16s first-segment render spike → lips start ~1s
  (was ~5s), long replies keep up. **Burst-feed** (`MUSETALK_FEED_BURST_S=1.0`) cut lip-start
  lag ~1.9s→~0.8s. Full writeup: `docs/PROBLEMS-AND-FIXES.md` P1/P2.
- **✅ FIXED — steady-mode voice screech.** `MUSETALK_SYNC_MODE=steady` (synced start, the
  default now) used to intermittently screech on long replies. Real cause: pipecat's output
  transport discarded a partial **odd-byte** audio buffer (on a >3s render-stall gap AND on the
  per-turn `TTSStoppedFrame`) → sample misalignment. Fixed by `musetalk_video._align_even` (every
  downstream frame kept whole-sample via a 1-byte carry) + `main.py::_relax_bot_vad_stop_timeout`.
  The dead frame-copy attempts + misleading per-chunk-RMS debug logs were removed. Full writeup +
  regression test (`scripts/_screech_repro_test.py`): `docs/PROBLEMS-AND-FIXES.md` P3.

## 2. (Optional) Make the WSL TTS setup durable
- The WSL IP changes on `wsl --shutdown`. Either switch WSL to **mirrored networking**
  (then `localhost:8001` works and `.env` never needs editing), or script the IP lookup
  into `run.ps1`.
- (Optional perf) the vLLM engine runs **eager** (no CUDA graphs). Enabling graphs needs
  more toolchain in WSL; could shave a bit more off TTFB.

## 3. (Optional) Push latency / quality further
- Try the LLM via Google's own API (Asia region) instead of OpenRouter (US) to shorten
  the network hop.
- `TTS_PROVIDER=elevenlabs` / `deepgram` are live fallbacks if CosyVoice ever misbehaves.

---

### Quick reference
- Run: `wsl ... run_vllm_server.sh` (TTS) → `.\scripts\run.ps1` (avatar + pipeline) → `/client/`
- Check imports: `python -m scripts.preflight`
- Revert TTS to the Windows PyTorch server: `.env COSYVOICE_URL=http://localhost:8001` + start it
- All settings in `.env`. Full status in `STATUS.md`; full workflow in `WORKFLOW.md`.
