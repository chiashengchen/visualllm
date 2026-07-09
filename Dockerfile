# Speech-only pipeline (no avatar, no GPU required).
# Runs: VAD → Deepgram STT → OpenRouter LLM → cloud TTS → WebRTC browser client.
# Set ENABLE_AVATAR=0 (the default here) to skip the MuseTalk stage.

FROM python:3.11-slim

# System deps for aiohttp / aiortc (OpenSSL, libvpx for VP8 encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    libvpx-dev \
    libopus-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the modules needed for the containerized pipeline
COPY pipeline/ ./pipeline/
COPY local_services/__init__.py ./local_services/
COPY local_services/cosyvoice_tts.py ./local_services/
COPY local_services/weather_chain_llm.py ./local_services/
COPY local_services/avatar_memory.py ./local_services/
COPY local_services/musetalk_video.py ./local_services/
COPY local_services/first_piece_aggregator.py ./local_services/
COPY local_services/sherpa_stt.py ./local_services/
COPY local_services/funasr_stt.py ./local_services/
COPY local_services/nimbus_client/ ./local_services/nimbus_client/
COPY log_setup.py .

# Excluded (Windows-only or GPU-only):
#   local_services/musetalk_server/   — GPU / conda env
#   local_services/moss_server/       — GPU / conda env
#   local_services/config_panel/      — Win32 TerminateProcess
#   local_services/musetalk_video.py  — avatar client
#   scripts/                          — PowerShell / Windows
#   *.exe, *.ps1, *.cs

# Cloud Run sets PORT; pipecat runner respects it via --port
ENV PORT=7860
ENV ENABLE_AVATAR=0
ENV AVATAR_MEMORY=0
ENV TTS_PROVIDER=deepgram
ENV LLM_PROVIDER=openrouter
# Disable ICE subnet restriction (Tailscale-specific, not needed in GCP)
ENV WEBRTC_ICE_SUBNET=0
# Cloud Run HTTP timeout is 60s by default; WebSocket sessions run longer.
# Set --timeout 3600 in the Cloud Run deploy command.
ENV BOT_VAD_STOP_FALLBACK_SECS=600

EXPOSE ${PORT}

CMD ["python", "-m", "pipeline.main", "--host", "0.0.0.0"]
