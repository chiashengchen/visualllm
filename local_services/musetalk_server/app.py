"""MuseTalk real-time lip-sync server (FastAPI + websocket).

Protocol (matches local_services/musetalk_video.py):
  client -> server:
    text json {"type":"config","fps":25}
    text json {"type":"speech_start"} / {"type":"speech_end"} / {"type":"reset"}
    binary: 16-bit PCM mono @16 kHz audio chunks (TTS audio, resampled by client)
  server -> client:
    binary: raw RGB frame buffers (IMAGE_SIZE*IMAGE_SIZE*3 bytes) at `fps`

Implementation notes
--------------------
This drives MuseTalk v1.5 locally on the GPU. It reuses the upstream model code
in ``vendor/MuseTalk`` (cloned next to this file) but replaces two things so it
runs on this machine:

* **Streaming** instead of whole-file inference. Audio arrives in PCM chunks; we
  buffer it and run UNet/VAE on fixed segments (``SEG_FRAMES`` frames each), so
  the avatar starts talking mid-utterance. Idle neutral frames keep the WebRTC
  video track alive between turns.
* **No mmpose/DWPose** for avatar preparation. Upstream uses DWPose (needs
  mmcv/mmpose, which require a CUDA compiler that isn't installed here). DWPose
  is only used to get the 68 iBUG face landmarks (keypoints[23:91]); we get the
  same 68 landmarks from ``face_alignment`` (pure-torch, pip-only) and feed them
  through MuseTalk's exact bbox math. Preparation is one-time and cached.

The realtime loop uses only the VAE decoder + UNet + Whisper — no landmark model
— so ``face_alignment`` is only imported during preparation.

Setup is handled by the project: a dedicated ``musetalk`` conda env (cu128 torch
+ diffusers/transformers + face_alignment) and weights under
``vendor/MuseTalk/models``. Run with:
    conda run -n musetalk python -m local_services.musetalk_server.app
VRAM: ~4-6 GB.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import asyncio
from pathlib import Path

# face_alignment 1.5 wraps its net in torch.compile, which needs Triton (absent
# on Windows). Disable TorchDynamo so it runs eagerly — MuseTalk isn't compiled.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

# --- paths -----------------------------------------------------------------
SERVER_DIR = Path(__file__).resolve().parent
MUSETALK_ROOT = SERVER_DIR / "vendor" / "MuseTalk"
MODELS_DIR = MUSETALK_ROOT / "models"
CACHE_DIR = SERVER_DIR / "avatar_cache"

# AVATAR_REF is given relative to the project root; resolve before we chdir.
AVATAR_REF = Path(os.getenv("AVATAR_REF", "assets/avatar.png")).resolve()

# --- knobs (env-overridable) ----------------------------------------------
IMAGE_SIZE = int(os.getenv("MUSETALK_SIZE", "512"))   # output frame is SIZE x SIZE
AUDIO_SR = 16000                                       # Whisper expects 16 kHz
DEFAULT_FPS = int(os.getenv("MUSETALK_FPS", "20"))  # ~32ms/frame GPU floor -> 20fps keeps realtime headroom
SEG_FRAMES = int(os.getenv("MUSETALK_SEG_FRAMES", "8"))   # frames per UNet segment
IDLE_FPS = int(os.getenv("MUSETALK_IDLE_FPS", "10"))       # neutral-frame rate between turns
BATCH_SIZE = int(os.getenv("MUSETALK_BATCH", "8"))
PAD_LEFT = int(os.getenv("MUSETALK_PAD_LEFT", "2"))
PAD_RIGHT = int(os.getenv("MUSETALK_PAD_RIGHT", "2"))
EXTRA_MARGIN = int(os.getenv("MUSETALK_EXTRA_MARGIN", "10"))
PARSING_MODE = os.getenv("MUSETALK_PARSING_MODE", "jaw")
# Cap the base-portrait resolution. Output is IMAGE_SIZE^2, so a huge source only
# slows the per-frame full-frame compositing (PIL). Keeps the face well above the
# 256px VAE crop while making blending realtime.
BASE_MAX = int(os.getenv("MUSETALK_BASE_MAX", "768"))
# How long the pump waits (queue empty + no audio) before deciding a `speech_end`
# was dropped and reverting to the idle loop. Must comfortably exceed the largest
# normal inter-chunk gap within one utterance, or it flips to idle mid-sentence.
IDLE_WATCHDOG_S = float(os.getenv("MUSETALK_IDLE_WATCHDOG_S", "3.0"))

app = FastAPI(title="MuseTalk realtime")

# The engine holds shared GPU model + per-turn cursor/fps state, so concurrent
# renders (even across connections) would corrupt each other. This server is
# single-client by design; the lock enforces one GPU inference at a time.
_render_lock = asyncio.Lock()

# Single-client session guard. A new /stream connection SUPERSEDES the prior one:
# we set the previous connection's `closed` event so its pump stops instead of two
# pumps competing (the reconnect-freeze: a left-open tab / a pipeline restart left a
# stale connection whose pump kept holding the line). Holds the active `closed` event.
_active_closed: "asyncio.Event | None" = None


class MuseTalkEngine:
    """MuseTalk models + a prepared (cached) avatar, with streaming inference."""

    def __init__(self, ref_path: Path, size: int, fps: int):
        self.ref_path = ref_path
        self.size = size
        self.fps = fps
        self._ready = False
        self.idx = 0  # base-frame cursor (cycles for video refs; static for a photo)
        self._trt = None  # set by _init_trt() when MUSETALK_TRT=1; None => PyTorch render path

    # --- one-time load ----------------------------------------------------
    def load(self):
        if self._ready:
            return
        if not MODELS_DIR.exists():
            raise RuntimeError(
                f"MuseTalk weights not found at {MODELS_DIR}. Run download_weights "
                f"(see local_services/musetalk_server/vendor/MuseTalk)."
            )

        import torch
        import cv2  # noqa: F401  (ensures cv2 import errors surface early)
        from transformers import WhisperModel

        # MuseTalk's checkpoints are legacy/pickled; PyTorch 2.6+ defaults
        # torch.load to weights_only=True and rejects them. Restore the old
        # behavior for the vendored loaders (trusted local weights).
        if not getattr(torch.load, "_musetalk_patched", False):
            _orig_load = torch.load

            def _load(*a, **k):
                k.setdefault("weights_only", False)
                return _orig_load(*a, **k)

            _load._musetalk_patched = True
            torch.load = _load

        # MuseTalk uses CWD-relative model paths; run from its root.
        os.chdir(MUSETALK_ROOT)
        if str(MUSETALK_ROOT) not in sys.path:
            sys.path.insert(0, str(MUSETALK_ROOT))

        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # NOTE: cudnn.benchmark was True ("fixed shapes"), but profiling proved the turn-START
        # segment has a DIFFERENT shape than mid-turn segments, so benchmark RE-AUTOTUNED the
        # first segment of EVERY turn -> ~16s GPU spike on this shared GPU (the late-start/stall).
        # False stops the per-turn re-tune; steady segments stay ~real-time. (allow TF32 kept.)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"Loading MuseTalk v1.5 models on {self.device} …")

        vae, unet, pe = load_all_model(
            unet_model_path=str(MODELS_DIR / "musetalkV15" / "unet.pth"),
            vae_type="sd-vae",
            unet_config=str(MODELS_DIR / "musetalkV15" / "musetalk.json"),
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = pe.half().to(self.device)
        vae.vae = vae.vae.half().to(self.device)
        unet.model = unet.model.half().to(self.device)
        self.vae, self.unet = vae, unet
        self.weight_dtype = unet.model.dtype

        self.audio_processor = AudioProcessor(feature_extractor_path=str(MODELS_DIR / "whisper"))
        self.feature_extractor = self.audio_processor.feature_extractor
        whisper = WhisperModel.from_pretrained(str(MODELS_DIR / "whisper"))
        self.whisper = whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

        self._prepare_avatar()
        self._neutral = self._frame_to_bytes(self.frame_cycle[0])
        self._build_idle_loop()
        self._ready = True
        self._warmup()   # pay the first-inference cost NOW so the first real turn isn't cold
        # VRAM trim: warmup ran dummy segments to pay one-time cuDNN/kernel/alloc costs;
        # that leaves reserved-but-unused blocks in PyTorch's caching allocator. Return
        # them to the driver so the idle footprint reflects the real working set (this is
        # where the measured ~8.7GB vs the model's ~4-6GB gap mostly hides). Best-effort:
        # an empty_cache failure must never block the server coming up.
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
                self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- cache release is best-effort
            logger.exception("empty_cache after warmup failed (non-fatal).")
        # Optional TensorRT render path. Engines are a prebuilt artifact (built offline,
        # ~7min); load them if present. ANY failure -> stay on the proven PyTorch path.
        if os.getenv("MUSETALK_TRT", "0").lower() in ("1", "true", "yes"):
            try:
                self._init_trt()
                logger.info("MuseTalk TRT engines loaded (MUSETALK_TRT=1).")
            except Exception:  # noqa: BLE001 -- TRT is best-effort; fall back to torch
                logger.exception("TRT init failed; using the PyTorch render path.")
                self._trt = None
        logger.info(
            f"MuseTalk ready. {len(self.frame_cycle)} base frame(s) prepared; "
            f"{len(self._idle_loop)} idle frame(s)."
        )

    def _init_trt(self):
        """Load the prebuilt UNet + VAE-decoder TRT engines. Build them offline with
        local_services/musetalk_server/trt_build.py (see docs/superpowers/plans/
        2026-06-30-musetalk-tensorrt.md). Raises if the engines are absent."""
        from .trt_runtime import TRTModule

        cache = SERVER_DIR / "trt_cache"
        unet_e, vae_e = cache / "unet.engine", cache / "vae.engine"
        if not (unet_e.exists() and vae_e.exists()):
            raise RuntimeError(
                f"TRT engines not found in {cache}; build them offline first."
            )
        self._trt = {
            "unet": TRTModule(str(unet_e), self.device),
            "vae": TRTModule(str(vae_e), self.device),
        }

    # --- avatar preparation (mmpose-free, cached) -------------------------
    def _avatar_key(self) -> str:
        st = self.ref_path.stat()
        raw = f"{self.ref_path}|{st.st_size}|{int(st.st_mtime)}|v15|m{EXTRA_MARGIN}|{PARSING_MODE}|b{BASE_MAX}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _prepare_avatar(self):
        import cv2

        if not self.ref_path.exists():
            raise RuntimeError(
                f"Avatar reference not found: {self.ref_path}. Put a front-facing "
                f"portrait at assets/avatar.png (see assets/README.md)."
            )

        cache = CACHE_DIR / self._avatar_key()
        if (cache / "materials.pkl").exists():
            logger.info(f"Loading cached avatar from {cache}")
            with open(cache / "materials.pkl", "rb") as f:
                mats = pickle.load(f)
            self.frame_cycle = mats["frames"]
            self.coord_cycle = mats["coords"]
            self.mask_cycle = mats["masks"]
            self.mask_coords_cycle = mats["mask_coords"]
            self.latent_cycle = self.torch.load(cache / "latents.pt", map_location=self.device)
            return

        logger.info("Preparing avatar (one-time): detecting landmarks + encoding latents …")
        from musetalk.utils.blending import get_image_prepare_material

        # Load reference as a list of BGR frames (image -> 1 frame; video -> N).
        frames = self._read_ref_frames(self.ref_path)
        coords = self._landmark_bboxes(frames)

        valid_frames, valid_coords, latents = [], [], []
        for bbox, frame in zip(coords, frames):
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])      # v15 extra margin
            bbox = (x1, y1, x2, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.vae.get_latents_for_unet(resized))
            valid_frames.append(frame)
            valid_coords.append(bbox)

        if not valid_frames:
            raise RuntimeError(
                "No face detected in the avatar reference. Use a clear, front-facing "
                "portrait at assets/avatar.png."
            )

        # Ping-pong the cycle so a short clip loops smoothly (a photo stays static).
        self.frame_cycle = valid_frames + valid_frames[::-1]
        self.coord_cycle = valid_coords + valid_coords[::-1]
        self.latent_cycle = latents + latents[::-1]

        self.mask_cycle, self.mask_coords_cycle = [], []
        for frame, bbox in zip(self.frame_cycle, self.coord_cycle):
            mask, crop_box = get_image_prepare_material(
                frame, list(bbox), fp=self.fp, mode=PARSING_MODE
            )
            self.mask_cycle.append(mask)
            self.mask_coords_cycle.append(crop_box)

        cache.mkdir(parents=True, exist_ok=True)
        with open(cache / "materials.pkl", "wb") as f:
            pickle.dump(
                {
                    "frames": self.frame_cycle,
                    "coords": self.coord_cycle,
                    "masks": self.mask_cycle,
                    "mask_coords": self.mask_coords_cycle,
                },
                f,
            )
        self.torch.save(self.latent_cycle, cache / "latents.pt")
        logger.info(f"Avatar materials cached to {cache}")

    def _read_ref_frames(self, path: Path):
        import cv2

        def _cap(fr):
            h, w = fr.shape[:2]
            m = max(h, w)
            if m > BASE_MAX:
                s = BASE_MAX / m
                fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            return fr

        ext = path.suffix.lower()
        if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(_cap(fr))
                if len(frames) >= 200:  # cap clip length to bound prep time/VRAM
                    break
            cap.release()
            if not frames:
                raise RuntimeError(f"Could not read frames from {path}")
            return frames
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read image {path}")
        return [_cap(img)]

    def _landmark_bboxes(self, frames):
        """68-landmark bbox per frame via face_alignment — MuseTalk's exact math,
        DWPose replaced. Returns a list of (x1,y1,x2,y2) or None."""
        import face_alignment

        lt = getattr(face_alignment.LandmarksType, "TWO_D",
                     getattr(face_alignment.LandmarksType, "_2D", None))
        fa = face_alignment.FaceAlignment(
            lt, flip_input=False,
            device="cuda" if self.device.type == "cuda" else "cpu",
        )

        out = []
        for frame in frames:
            rgb = frame[:, :, ::-1]  # BGR -> RGB for face_alignment
            preds = fa.get_landmarks(rgb)
            if not preds:
                out.append(None)
                continue
            lm = preds[0].astype(np.int32)                       # (68,2) iBUG-68
            half_y = lm[29][1]                                   # nose bridge (idx 29)
            half_dist = int(np.max(lm[:, 1]) - half_y)
            upper = max(0, half_y - half_dist)
            x1, x2 = int(np.min(lm[:, 0])), int(np.max(lm[:, 0]))
            y1, y2 = int(upper), int(np.max(lm[:, 1]))
            if x1 < 0 or x2 - x1 <= 0 or y2 - y1 <= 0:
                out.append(None)
            else:
                out.append((x1, y1, x2, y2))
        del fa  # free the landmark model; the realtime loop never needs it
        return out

    def _warmup(self):
        """Run a couple of dummy segments through the full render path at load time.

        The FIRST real inference otherwise pays one-time costs -- cuDNN autotune
        (benchmark.True), CUDA kernel compilation, lazy allocations -- which made the
        first turn after a server start render far below fps (the "cold first-turn stall":
        audio finishes, the avatar plays its backlog out for seconds afterward). Doing it
        here, on silent audio we discard, means the user's first real turn is already warm.
        Best-effort: a warmup failure must NEVER block the server from coming up.
        """
        import time as _t
        try:
            seg = np.zeros(self.samples_for_frames(SEG_FRAMES), dtype=np.float32)   # ~SEG_FRAMES of silence
            t0 = _t.time()
            n = 0
            for _ in range(2):   # 1st call compiles/autotunes; 2nd confirms the warm path
                n = len(self.render_segment(seg))
            self.idx = 0   # undo the cursor advance so real turns start from the rest pose
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            logger.info(f"MuseTalk warmup done in {_t.time()-t0:0.1f}s ({n} frames/segment).")
        except Exception:  # noqa: BLE001 -- warmup is best-effort; never block startup
            logger.exception("MuseTalk warmup failed (non-fatal; first turn may be cold).")
            self.idx = 0

    # --- realtime inference ----------------------------------------------
    def samples_per_frame(self, fps: int) -> int:
        return int(AUDIO_SR / fps)

    def samples_for_frames(self, n_frames: int, fps: int | None = None) -> int:
        """Samples that render to EXACTLY n_frames lip frames.

        The renderer counts frames as floor(len/sr*fps) (get_whisper_chunk). Sizing a
        segment as int(sr/fps)*n (the old way) TRUNCATES sr/fps -- at fps that don't
        divide 16000 (e.g. 12: sr/fps=1333.33->1333) an 8-frame segment lands at
        floor(7.998)=7, losing one lip frame PER segment. Over a long reply that ~12.5%
        deficit accumulates and the avatar finishes ~1-2s before the audio. Sizing to the
        frame's UPPER sample boundary (ceil) keeps floor() at exactly n_frames for any fps
        (it was only ever correct at fps that divide 16000, e.g. the old default 20)."""
        f = int(fps if fps is not None else self.fps)
        return math.ceil(n_frames * AUDIO_SR / f)

    def reset_idx(self):
        self.idx = 0

    def neutral_frame(self) -> bytes:
        return self._neutral

    def idle_frames(self) -> list[bytes]:
        """RGB frames the pump plays between turns so the face stays alive
        instead of freezing on the neutral portrait. See _build_idle_loop."""
        return getattr(self, "_idle_loop", [])

    def _build_idle_loop(self) -> None:
        """Precompute a seamless idle-motion loop.

        MuseTalk only moves the *mouth* (driven by audio), so on a still photo
        there is nothing to animate between turns -- the avatar freezes. To keep
        it looking alive we synthesize a gentle "breathing" loop from the neutral
        portrait: a slow scale pulse + a slower micro head-sway (rotation +
        translate), anchored low so it reads as the chest/shoulders breathing.

        If the avatar reference is a *video* (frame_cycle has real motion), we
        loop those real frames instead -- natural breathing and blinks.

        Disable with MUSETALK_IDLE_MOTION=0.
        """
        import cv2

        if os.getenv("MUSETALK_IDLE_MOTION", "1").lower() not in ("1", "true", "yes"):
            self._idle_loop = []
            return

        # Video reference (>2 frames after the ping-pong): use the real motion.
        if len(self.frame_cycle) > 2:
            self._idle_loop = [self._frame_to_bytes(f) for f in self.frame_cycle]
            return

        base = cv2.resize(
            self.frame_cycle[0], (self.size, self.size), interpolation=cv2.INTER_AREA
        )
        h, w = base.shape[:2]
        n = max(8, int(os.getenv("MUSETALK_IDLE_FRAMES", "80")))  # loop length
        # Overall amplitude multiplier -- bump MUSETALK_IDLE_GAIN if the breathing
        # is too subtle to notice (or down to calm it). Base values are tuned to
        # be clearly visible without looking like camera shake.
        g = float(os.getenv("MUSETALK_IDLE_GAIN", "1.0"))
        anchor = (w / 2.0, h * 0.92)  # low anchor -> breathing, not bobbing head
        loop: list[bytes] = []
        for t in range(n):
            ph = 2.0 * np.pi * t / n
            scale = 1.0 + 0.015 * g * np.sin(ph)        # breathing pulse (~1.5%)
            dy = 4.0 * g * np.sin(ph)                    # vertical bob (px)
            ang = 0.9 * g * np.sin(0.5 * ph)             # slow head sway (deg)
            dx = 3.0 * g * np.sin(0.5 * ph)              # slow horizontal drift (px)
            M = cv2.getRotationMatrix2D(anchor, ang, scale)
            M[0, 2] += dx
            M[1, 2] += dy
            warp = cv2.warpAffine(
                base, M, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            rgb = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
            loop.append(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self._idle_loop = loop

    def _frame_to_bytes(self, frame_bgr) -> bytes:
        import cv2

        out = cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()

    def _audio_features(self, audio: np.ndarray):
        seg_len = 30 * AUDIO_SR
        segments = [audio[i:i + seg_len] for i in range(0, len(audio), seg_len)] or [audio]
        feats = []
        for seg in segments:
            af = self.feature_extractor(
                seg, return_tensors="pt", sampling_rate=AUDIO_SR
            ).input_features.to(self.weight_dtype)
            feats.append(af)
        return feats, len(audio)

    def _composite(self, res_bgr: np.ndarray, idx: int) -> bytes:
        import cv2
        from musetalk.utils.blending import get_image_blending

        bbox = self.coord_cycle[idx]
        x1, y1, x2, y2 = bbox
        ori = self.frame_cycle[idx].copy()
        face = cv2.resize(res_bgr.astype(np.uint8), (x2 - x1, y2 - y1))
        combine = get_image_blending(
            ori, face, bbox, self.mask_cycle[idx], self.mask_coords_cycle[idx]
        )
        return self._frame_to_bytes(combine)

    def render_segment(self, audio: np.ndarray) -> list[bytes]:
        """One audio segment (float32 [-1,1] @16k) -> list of RGB frame buffers.

        Runs on a worker thread (GPU-bound). Index cursor and base frames are
        cycled per produced frame so latents and composites stay aligned.
        """
        torch = self.torch
        prof = os.getenv("MUSETALK_PROFILE")
        import time as _t

        t0 = _t.time()
        try:
            feats, length = self._audio_features(audio)
            t_feat = _t.time()
            chunks = self.audio_processor.get_whisper_chunk(
                feats, self.device, self.weight_dtype, self.whisper, length,
                fps=self.fps, audio_padding_length_left=PAD_LEFT,
                audio_padding_length_right=PAD_RIGHT,
            )
            t_whisper = _t.time()
        except (AssertionError, SystemExit, Exception):  # noqa: BLE001
            logger.exception("whisper chunking failed; dropping segment")
            return []

        L = len(self.latent_cycle)
        out: list[bytes] = []
        gpu_s = 0.0
        comp_s = 0.0
        with torch.no_grad():
            for i in range(0, len(chunks), BATCH_SIZE):
                w_batch = chunks[i:i + BATCH_SIZE].to(self.device)
                n = w_batch.shape[0]
                idxs = [(self.idx + k) % L for k in range(n)]
                latent_batch = torch.cat([self.latent_cycle[x] for x in idxs], dim=0).to(
                    device=self.device, dtype=self.unet.model.dtype
                )
                audio_feat = self.pe(w_batch)
                if self._trt is not None:
                    # TRT path: engines replace the UNet + VAE-decoder GPU calls. The pre/post
                    # math MUST match VAE.decode_latents exactly so _composite is unchanged:
                    #   decode_latents = (1/sf)*latents -> vae.decode -> /2+0.5 clamp ->
                    #                    permute(0,2,3,1) -> *255 uint8 -> [...,::-1] (BGR)
                    sample = self._trt["unet"](
                        latent=latent_batch, timestep=self.timesteps, audio=audio_feat
                    )["sample"]
                    dec_in = (1.0 / self.vae.scaling_factor) * sample.to(self.vae.vae.dtype)
                    img = self._trt["vae"](latent=dec_in)["image"]   # (n,3,256,256) raw decode
                    img = (img / 2 + 0.5).clamp(0, 1)
                    recon = (
                        img.permute(0, 2, 3, 1).float().cpu().numpy() * 255
                    ).round().astype("uint8")[..., ::-1]
                else:
                    pred = self.unet.model(
                        latent_batch, self.timesteps, encoder_hidden_states=audio_feat
                    ).sample
                    pred = pred.to(dtype=self.vae.vae.dtype)
                    recon = self.vae.decode_latents(pred)  # [n,256,256,3] BGR uint8
                tg = _t.time()
                gpu_s += tg - (t_whisper if i == 0 else tc)
                for k in range(n):
                    out.append(self._composite(recon[k], idxs[k]))
                tc = _t.time()
                comp_s += tc - tg
                self.idx = (self.idx + n) % L
        if prof:
            logger.info(
                f"[profile] feat={1000*(t_feat-t0):.0f}ms "
                f"whisper={1000*(t_whisper-t_feat):.0f}ms "
                f"gpu={1000*gpu_s:.0f}ms composite={1000*comp_s:.0f}ms "
                f"-> {len(out)} frames"
            )
        return out


engine = MuseTalkEngine(AVATAR_REF, IMAGE_SIZE, DEFAULT_FPS)


@app.on_event("startup")
def _startup():
    engine.load()


@app.websocket("/stream")
async def stream(ws: WebSocket):
    global _active_closed
    await ws.accept()
    # Session guard: supersede any prior connection so a reconnect can't leave two
    # pumps competing (the freeze). Signal the previous connection's pump to stop.
    if _active_closed is not None:
        _active_closed.set()
    fps = engine.fps
    seg_samples = engine.samples_for_frames(SEG_FRAMES, fps)   # ceil-sized: exactly SEG_FRAMES/seg
    audio_buf = np.zeros(0, dtype=np.float32)
    # Bounded queue of rendered frames; the pump drains it at a STEADY fps. A SMALLER
    # cap is the documented SAFE lag lever for live mode: under GPU contention the render
    # skips stale frames instead of letting the lips fall arbitrarily far behind the voice.
    # Do NOT re-lock the voice to video (that froze it). 600 ~= 30s @20fps (effectively
    # unbounded); tighten via MUSETALK_OUT_Q for a shorter max trail.
    out_q: asyncio.Queue = asyncio.Queue(maxsize=int(os.getenv("MUSETALK_OUT_Q", "600")))
    closed = asyncio.Event()
    _active_closed = closed
    loop = asyncio.get_event_loop()
    speaking = asyncio.Event()   # set while a turn's audio is being rendered
    st = {"last_audio": 0.0}     # loop.time() of the last audio chunk (watchdog)
    # A/V-sync markers: the pump tells the client which frames are REAL lip-synced
    # render (vs idle) and how many have played, so the client paces the voice to the
    # actually-rendered video (see local_services/musetalk_video.py). Best-effort.
    idle_grace = float(os.getenv("MUSETALK_IDLE_GRACE", "0.3"))
    # Readiness prime: frames to buffer before a turn starts playing (the "wait until ready"
    # cushion). ~6 @12fps = 0.5s -> absorbs render hiccups so the locked voice never stutters.
    lead_frames = int(os.getenv("MUSETALK_LEAD_FRAMES", "6"))

    async def _mark(payload: dict) -> None:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:  # noqa: BLE001 -- markers are best-effort
            pass

    async def pump():
        """Emit a steady `fps` video stream. WebRTC/mobile decoders freeze on
        bursty input, so we pace output: send the next rendered frame if one is
        ready. When idle (between turns) we play a gentle breathing loop so the
        face stays alive instead of freezing on the neutral portrait; while
        speaking we hold the last frame if rendering momentarily lags."""
        interval = 1.0 / max(1, fps)
        idle = engine.idle_frames()
        idle_n = len(idle)
        idle_i = 0
        last = idle[0] if idle_n else engine.neutral_frame()
        was_speaking = False
        nxt = loop.time()
        # Marker state, driven by REAL frames drained (not idle/held): `playing` spans a
        # turn's rendered frames; `real_sent` counts only truly-dequeued frames so a render
        # stall stops the client's voice with it.
        playing = False
        real_sent = 0
        last_clock = 0
        empty_since = None
        try:
            while not closed.is_set():
                sp = speaking.is_set()
                # Watchdog: if a speech_end was dropped, don't stay "speaking"
                # forever -- once the queue drains and audio has stopped, idle.
                if sp and out_q.empty() and (loop.time() - st["last_audio"]) > IDLE_WATCHDOG_S:
                    speaking.clear()
                    sp = False
                if was_speaking and not sp:
                    idle_i = 0  # resume idle from the rest pose (seamless)
                was_speaking = sp

                # READINESS PRIME: at the start of a turn, don't begin playing until a small
                # LEAD of frames is buffered (the "wait until the avatar is ready" gate). This
                # gives the locked audio a cushion so a render hiccup is absorbed (no stutter)
                # and the voice starts in step with a ready avatar. Hold the last frame while priming.
                if not playing and sp and out_q.qsize() < lead_frames:
                    await ws.send_bytes(last)
                    nxt += interval
                    await asyncio.sleep(max(0.0, nxt - loop.time()))
                    continue

                got = None
                try:
                    got = out_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                if got is not None:
                    last = got
                    if not playing:                     # primed & ready: start the clock
                        playing = True
                        real_sent = 0
                        last_clock = 0
                        await _mark({"type": "video_start"})
                    real_sent += 1
                    empty_since = None
                    if real_sent - last_clock >= 2:     # ~every 2 real frames
                        last_clock = real_sent
                        await _mark({"type": "video_clock", "frames": real_sent})
                else:
                    if playing:
                        now = loop.time()
                        if empty_since is None:
                            empty_since = now
                        # Only END the turn's video segment when the turn is REALLY over
                        # (speech_end -> not sp) AND the queue has drained -- not on a brief
                        # mid-turn render underflow (that would re-segment + desync the client).
                        elif not sp and now - empty_since >= idle_grace:
                            playing = False
                            empty_since = None
                            await _mark({"type": "video_end"})
                        # else: render hiccup OR end-of-turn wait -> HOLD `last`. The stream MUST
                        # stay continuous (a gap = the avatar freezes); the client caps its own
                        # release against the audio so a held frame can't run video ahead.
                    elif not sp and idle_n:
                        last = idle[idle_i % idle_n]
                        idle_i += 1
                    elif not sp:
                        # No idle loop (MUSETALK_IDLE_MOTION=0): settle to the NEUTRAL rest pose
                        # between turns. Without this `last` stays on the last drained frame -- which
                        # with MUSETALK_END_TAIL_FRAMES=0 is the last SPOKEN frame (parted mouth), so
                        # the face would rest parted AND the client's close crossfade would cache that
                        # as its target (a no-op). END_TAIL>0 used to hide this by draining a neutral.
                        last = engine.neutral_frame()
                # Always emit a frame this tick (real, held-last, or idle) so the video never
                # freezes -- the end-of-turn freeze was caused by skipping the send on underflow.
                await ws.send_bytes(last)
                nxt += interval
                await asyncio.sleep(max(0.0, nxt - loop.time()))
        except Exception:  # noqa: BLE001
            pass

    def enqueue(frame: bytes) -> None:
        # Stay realtime: drop the oldest frame rather than lag behind (or, for the
        # closing neutral frame, rather than raise QueueFull and kill the socket).
        if out_q.full():
            try:
                out_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        out_q.put_nowait(frame)

    async def render(segment: np.ndarray) -> int:
        # One GPU inference at a time across the whole process (shared engine).
        async with _render_lock:
            frames = await asyncio.to_thread(engine.render_segment, segment)
        for f in frames:
            enqueue(f)
        return len(frames)

    pump_task = asyncio.create_task(pump())
    turn_frames = 0

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                evt = json.loads(msg["text"])
                kind = evt.get("type")
                if kind == "config":
                    fps = int(evt.get("fps", fps))
                    engine.fps = fps
                    seg_samples = engine.samples_for_frames(SEG_FRAMES, fps)
                    logger.info(f"[stream] config: fps={fps}")
                elif kind == "speech_start":
                    turn_frames = 0
                    speaking.set()                 # pump holds, not idle, now
                    st["last_audio"] = loop.time()
                elif kind in ("reset", "speech_end"):
                    # Render the FINAL audio even if it's less than a full segment (pad up to one
                    # frame) so the last word/syllable isn't dropped -- the "cut at the end".
                    if kind == "speech_end" and len(audio_buf) > 0:
                        # Pad the trailing remainder UP to a whole frame's worth so the last
                        # partial frame renders instead of being floor()'d away (the dropped
                        # final syllable). Use the SAME ceil sizing as the main loop so the
                        # frame count stays = audio_seconds*fps end-to-end (no early finish).
                        f_final = max(1, math.ceil(len(audio_buf) / AUDIO_SR * fps))
                        seg_len = engine.samples_for_frames(f_final, fps)
                        pad = max(0, seg_len - len(audio_buf))
                        seg = (
                            np.concatenate([audio_buf, np.zeros(pad, np.float32)])
                            if pad else audio_buf
                        )
                        turn_frames += await render(seg)
                    audio_buf = np.zeros(0, dtype=np.float32)
                    engine.reset_idx()
                    speaking.clear()               # let the idle loop resume
                    if kind == "speech_end":
                        logger.info(f"[stream] turn rendered {turn_frames} frames")
                        # Graceful TAIL: a few neutral frames so the avatar eases out instead of
                        # snapping shut. Set to 0 when the CLIENT runs the close crossfade
                        # (MUSETALK_CLOSE_FADE_FRAMES) -- then the last buffered frame must stay the
                        # last SPOKEN frame, so we must NOT append any neutral tail (allow a true 0).
                        tail = int(os.getenv("MUSETALK_END_TAIL_FRAMES", "4"))
                        for _ in range(max(0, tail)):
                            enqueue(engine.neutral_frame())
                continue

            data = msg.get("bytes")
            if not data:
                continue
            speaking.set()                          # defensive: audio implies a turn
            st["last_audio"] = loop.time()
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buf = np.concatenate([audio_buf, pcm])
            while len(audio_buf) >= seg_samples:
                seg = audio_buf[:seg_samples]
                audio_buf = audio_buf[seg_samples:]
                turn_frames += await render(seg)

    except WebSocketDisconnect:
        logger.info("MuseTalk client disconnected.")
    except Exception:  # noqa: BLE001
        logger.exception("MuseTalk stream error")
    finally:
        closed.set()
        await asyncio.gather(pump_task, return_exceptions=True)


@app.get("/health")
def health():
    return {"ok": engine._ready, "avatar": str(AVATAR_REF), "size": IMAGE_SIZE}


if __name__ == "__main__":
    # ws_ping_interval/timeout=None: the stream pump sends frames continuously at
    # `fps`, so the socket is never idle. The websockets keepalive ping adds no
    # value here and its write races the pump's high-rate send_bytes -- that race
    # is what was dropping the connection after a turn or two (AssertionError in
    # websockets' keepalive_ping). Disable it; the steady frame flow IS the
    # liveness signal.
    uvicorn.run(
        app, host="0.0.0.0", port=8002,
        ws_ping_interval=None, ws_ping_timeout=None,
    )
