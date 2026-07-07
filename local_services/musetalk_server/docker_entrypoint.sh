#!/bin/bash
# MuseTalk server entrypoint: download weights into the models volume on first
# run (mirrors upstream download_weights.sh, minus dwpose/syncnet/v1.0 which the
# realtime server never loads), then start the FastAPI ws server.
set -e

MODELS=/app/local_services/musetalk_server/vendor/MuseTalk/models

# Check every required file, not just unet.pth — a partial first download
# (e.g. gdown failing) must not be mistaken for "weights ready".
if [ ! -f "$MODELS/musetalkV15/unet.pth" ]; then
  echo "[entrypoint] downloading MuseTalk core weights into $MODELS ..."
  huggingface-cli download TMElyralab/MuseTalk --local-dir "$MODELS" \
    --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
  huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "$MODELS/sd-vae" \
    --include "config.json" "diffusion_pytorch_model.bin"
  huggingface-cli download openai/whisper-tiny --local-dir "$MODELS/whisper" \
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
fi
mkdir -p "$MODELS/face-parse-bisent"
if [ ! -f "$MODELS/face-parse-bisent/79999_iter.pth" ]; then
  # positional arg: gdown >=5 removed the --id flag
  gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$MODELS/face-parse-bisent/79999_iter.pth"
fi
if [ ! -f "$MODELS/face-parse-bisent/resnet18-5c106cde.pth" ]; then
  curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -o "$MODELS/face-parse-bisent/resnet18-5c106cde.pth"
fi
echo "[entrypoint] weights ready."

# The portrait must be provided (mount it into the container at AVATAR_REF).
REF="${AVATAR_REF:-assets/avatar.png}"
if [ ! -f "/app/$REF" ] && [ ! -f "$REF" ]; then
  echo "[entrypoint] ERROR: no portrait at $REF — mount an image there" >&2
  exit 1
fi

cd /app
exec python -u -m local_services.musetalk_server.app
