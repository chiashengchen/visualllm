#!/usr/bin/env bash
# VM setup script for VisualLLM on GCP (Ubuntu 22.04 + T4/L4)
# Run once after VM is created:
#   gcloud compute ssh visualllm-gpu --zone=asia-northeast3-c --project=visualllm-prod \
#     -- 'bash -s' < scripts/vm_setup.sh

set -euo pipefail

echo "=== VisualLLM VM Setup ==="
echo "$(date)"

# ── 1. System deps ────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    git curl wget htop tmux unzip \
    ca-certificates gnupg lsb-release

# ── 2. Docker ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/5] Installing Docker..."
if command -v docker &>/dev/null; then
    echo "Docker already installed: $(docker --version)"
else
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor -o /usr/share/keyrings/docker.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo systemctl enable docker
    sudo systemctl start docker
    # Allow current user to run docker without sudo
    sudo usermod -aG docker "$USER"
    echo "Docker installed: $(docker --version)"
fi

# ── 3. NVIDIA Container Toolkit ───────────────────────────────────────────────
echo ""
echo ">>> [3/5] Installing nvidia-container-toolkit..."
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "nvidia-container-toolkit already installed"
else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#" \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "nvidia-container-toolkit installed"
fi

# Verify GPU is accessible from Docker
echo "GPU Docker test:"
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
    | grep "Tesla T4\|L4" || echo "WARNING: GPU not visible in Docker"

# ── 4. Clone repos ────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/5] Cloning repos..."
cd ~

if [ ! -d "visualllm" ]; then
    git clone https://github.com/chiashengchen/visualllm.git
    echo "Cloned visualllm"
else
    cd visualllm && git pull && cd ~
    echo "Updated visualllm"
fi

if [ ! -d "CosyVoice" ]; then
    git clone https://github.com/chiashengchen/CosyVoice.git
    echo "Cloned CosyVoice"
else
    cd CosyVoice && git pull && cd ~
    echo "Updated CosyVoice"
fi

# ── 5. .env file ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [5/5] Setting up .env..."
cd ~/visualllm

if [ ! -f ".env" ]; then
    # Pull API keys from GCP Secret Manager
    DEEPGRAM_KEY=$(gcloud secrets versions access latest \
        --secret=deepgram-key --project=visualllm-prod 2>/dev/null || echo "")
    OPENROUTER_KEY=$(gcloud secrets versions access latest \
        --secret=openrouter-key --project=visualllm-prod 2>/dev/null || echo "")

    cat > .env <<EOF
DEEPGRAM_API_KEY=${DEEPGRAM_KEY}
OPENROUTER_API_KEY=${OPENROUTER_KEY}
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
LANGUAGE=zh
EOF
    echo ".env created (keys pulled from Secret Manager)"

    if [ -z "$DEEPGRAM_KEY" ] || [ -z "$OPENROUTER_KEY" ]; then
        echo ""
        echo "WARNING: One or more API keys are empty."
        echo "  Edit ~/visualllm/.env and add the missing keys manually."
    fi
else
    echo ".env already exists, skipping"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete! ==="
echo ""
echo "IMPORTANT: Log out and back in for Docker group to take effect, then:"
echo ""
echo "  cd ~/visualllm"
echo "  docker compose build     # build images (~10 min first time)"
echo "  docker compose up        # start pipeline + CosyVoice"
echo ""
echo "  Pipeline: http://$(curl -s ifconfig.me 2>/dev/null || echo '<VM_IP>'):7860/client/"
