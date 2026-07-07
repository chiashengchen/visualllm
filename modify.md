# Modify Log

## 目標

把 visualllm 的 speech pipeline（STT → LLM → TTS → speech）部署到 GCP，跑在 T4 GPU VM 上，測試是否能降低延遲（特別是中文 TTS TTFB）。

---

## 主要改動

### 1. Pipeline — Avatar 可選 (`ENABLE_AVATAR` flag)

**檔案：** `pipeline/config.py`, `pipeline/main.py`

- 新增 `enable_avatar` config flag（預設 `1`，設 `0` 跳過 MuseTalk）
- `main.py` 根據 flag 決定是否加入 avatar stage，`video_out_enabled` 也一起跟著
- GCP 部署時設 `ENABLE_AVATAR=0`，跑純語音 pipeline，不需要 GPU 的 MuseTalk

---

### 2. GCP 環境建置

**檔案：** `scripts/gcp_setup.sh`, `scripts/vm_setup.sh`

- `gcp_setup.sh`：一次性設定 GCP project（啟用 API、建 Artifact Registry、設 Workload Identity Federation、存 API keys 到 Secret Manager）
- `vm_setup.sh`：VM 開機後的 bootstrap（裝 Docker、nvidia-container-toolkit、clone repos、設 .env）
- VM：`n1-standard-4` + T4 16GB，`asia-northeast3-c`（首爾），IP `8.230.16.27`

---

### 3. Docker 化

**檔案：** `Dockerfile`（pipeline）, `docker-compose.yml`

- `Dockerfile`：pipeline 的 image（python:3.11-slim，裝 pipecat + 相關 deps，`ENABLE_AVATAR=0`）
- `docker-compose.yml`：兩個 service
  - `cosyvoice`：TTS server，GPU，port 8001，healthcheck，weights volume
  - `pipeline`：Pipecat pipeline，CPU，port 7860，等 cosyvoice healthy 才起

---

### 4. CosyVoice TTS Server

**Repo：** `chiashengchen/CosyVoice`（fork）

**檔案：** `tts_stream_server.py`, `Dockerfile.tts`, `.dockerignore`

- `tts_stream_server.py`：FastAPI server，實作 visualllm pipeline 期望的 `POST /tts/stream` 介面（回傳 raw 16-bit PCM mono）
- `Dockerfile.tts`：CosyVoice 的 GPU image（CUDA 12.4 base、conda env python 3.10 + pynini、裝所有 Python deps）

**Dockerfile 修 bug 過程（一連串 build 錯誤）：**

| 問題 | 原因 | 修法 |
|------|------|------|
| `pkg_resources` not found | conda env 的 setuptools 太新（≥72 沒有 `pkg_resources`） | conda create 時加 `"setuptools<72"` |
| `numpy` not found（pyworld build） | pyworld 的 `setup.py` import numpy，但 numpy 還沒裝 | 先單獨 `pip install numpy==1.26.4` |
| tensorrt `wheel_stub` 錯誤 | tensorrt 的 pyproject.toml 用 `wheel_stub` build backend，環境沒有 | 過濾掉 requirements.txt 裡的 `tensorrt*`（`load_trt=False` 不需要） |
| vllm 與 torch 2.3.1 衝突 | 新版 vllm 要求 torch 2.4+，pip resolver 無限 backtrack | 移除 pip vllm，改 `USE_VLLM=0`（標準 PyTorch 推理） |
| `vllm_gpu_memory_utilization` 參數錯誤 | 此 fork 的 `CosyVoice2.__init__` 沒有這個參數 | 從 `tts_stream_server.py` 移除 |
| `No module named 'matcha'` | Matcha-TTS 是 git submodule，Docker context 沒帶進去 | 改用 pip 從 pinned commit 直接安裝：`matcha-tts @ git+https://...@dd9105b` |
| `torch.library.register_fake` 不存在 | `transformers==4.51.3` 需要 torch 2.4+，但 requirements 鎖 `torch==2.3.1` | 在 requirements 處理時 sed 替換成 `transformers==4.44.2` |

---

### 5. GitHub CI/CD

**檔案：** `.github/workflows/deploy.yml`

- `preflight` job：每次 push/PR 都跑，驗證 pipecat import 沒壞
- `build-deploy` job：main branch，用 Workload Identity Federation（無 key）auth 到 GCP，build + push image 到 Artifact Registry

---

### 6. MCP Testing Server

**目錄：** `testing_mcp/`

- `server.py`：FastMCP server，expose 5 個 tool
- `tools/tts_eval.py`：TTS 品質評估（round-trip CER：合成 → Deepgram STT → 比較原文）
- `tools/stt_eval.py`：STT 準確度評估（WER/CER）
- `tools/e2e_latency.py`：端到端延遲量測
- `tools/gcp_jobs.py`：觸發 Vertex AI 訓練 job、查 job 狀態、deploy model

---

## 目前狀態

CosyVoice image 正在 build（build4，`transformers==4.44.2` 修正後），build 完後會啟動 `docker compose up`，測試 `http://8.230.16.27:7860/client/`。
