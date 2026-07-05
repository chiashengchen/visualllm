# GCP 部署進度紀錄

_更新：2026-07-05。這份文件記錄 GCP VM（無 avatar）部署的現狀。本機 Windows + avatar 的狀態仍以 `STATUS.md` 為準。_

## 現狀：✅ 可用

`https://8.230.16.27/client/` — 語音對話（STT → LLM → TTS）端到端正常，
瀏覽器需點過自簽憑證警告（進階 → 繼續前往）。

## 版本（git SHA）

| Repo | Commit | 說明 |
|------|--------|------|
| visualllm | `c2f9fec` | fix: pipeline 改 host network，WebRTC UDP 繞過 Docker NAT |
| CosyVoice | `daa69db` | fix: `inference_zero_shot` 改傳 wav 檔路徑（原本傳 tensor 會 TypeError） |

VM 上兩個 repo（`/home/cschen/visualllm`、`/home/cschen/CosyVoice`）都已同步到上列 commit。

## 部署架構

- **VM**: `visualllm-gpu`，zone `asia-northeast3-c`，外部 IP `8.230.16.27`
- **docker compose** 三個服務：
  - `cosyvoice` — CosyVoice2-0.5B TTS（`USE_VLLM=0`，PyTorch 推理），GPU，port 8001
  - `pipeline` — Pipecat 語音管線，**`network_mode: host`**（WebRTC UDP 必須直接綁 host IP，Docker bridge NAT 會擋掉瀏覽器進來的 UDP），port 7860
  - `nginx` — HTTPS 終端（自簽憑證），443 → `host.docker.internal:7860`
- **STUN**: `pipeline/main.py::_configure_stun_servers()` 注入 `stun:stun.l.google.com:19302`，
  讓 aiortc 發現 GCP 1:1 NAT 後面的公網 IP（`WEBRTC_STUN_URL=0` 可關）
- **防火牆**（tag `visualllm`）: tcp 443/7860/8001/8002 + **udp 49152-65535**（WebRTC media）
- 設定: `LANGUAGE=zh`、LLM=openrouter（`google/gemini-2.5-flash-lite`）、STT=Deepgram nova-2

## Cloud Run CD — 已移除

`.github/workflows/deploy.yml` 原本會把 pipeline 部署到 Cloud Run（Deepgram TTS 版）。
現在改為 GCE VM 部署後已無用，**build-deploy job 已刪除**，只保留 preflight import 檢查（CI）。
GCP 上沒有留下任何 Cloud Run 服務（`gcloud run services list` = 0），不會產生費用。
VM 端更新方式：`git pull && sudo docker compose build && sudo docker compose up -d`（手動）。

## 硬體規格

| 項目 | 規格 |
|------|------|
| Machine type | `n1-standard-4` |
| CPU | 4 vCPU（Intel Xeon @ 2.00GHz） |
| RAM | 14 GiB |
| GPU | NVIDIA Tesla T4（16 GB VRAM） |
| Disk | 97 GB（已用 55 GB，主要是 26 GB 的 cosyvoice image） |

## 延遲實測（2026-07-05，中文短句）

| 指標 | 數值 | 備註 |
|------|------|------|
| **TTFO**（使用者停止說話 → bot 開始出聲） | **2.4 – 3.8 s**（median 3.8s，target 8s ✅） | pipeline `TtfoMeter` |
| LLM TTFB | 0.47 – 0.69 s | OpenRouter gemini-2.5-flash-lite |
| TTS 首個音訊 chunk（第一句） | 1.4 – 3.2 s | 短句快、長句慢 |
| TTS 首個音訊 chunk（後續句，GPU 忙碌時） | 最高 5.8 s | 前一句還在合成時排隊 |
| TTS 合成速度（RTF） | ~1.0 | 5.3 秒音訊耗時 ~5.5 秒。T4 上剛好即時，長回覆句間可能出現空隙 |

VAD 使用 Silero（本地、幾十 ms 級），TTFO 已含 VAD + STT + LLM + TTS 全鏈路。

## 資源用量

**閒置（連線待機）：**

| 服務 | CPU | RAM |
|------|-----|-----|
| cosyvoice | ~0.2% | **5.6 GiB** |
| pipeline | ~0.1% | 210 MiB |
| nginx | ~0% | 3 MiB |
| GPU VRAM | — | 3.3 GiB（模型常駐） |

**TTS 推理中：** GPU util 40–45%、VRAM 峰值 ~4.6 GiB。CPU 端 STT/LLM 都是雲端 API，本機負載很輕。

結論：T4 16GB 綽綽有餘（僅用 ~29%），瓶頸是 **TTS RTF ~1.0**（合成速度只勉強跟上播放速度），
不是記憶體。若要更快可考慮 `USE_VLLM=1`（本機 WSL 版實測 TTFB 3.4s→1.1s）或換 L4 GPU。

## 已知問題 / 待辦

- [ ] 自簽憑證：每個新瀏覽器都要手動信任。可改 Let's Encrypt（需要一個網域）
- [ ] 中文長回覆句間偶有停頓（TTS RTF ~1.0，T4 極限）；可試 `USE_VLLM=1`
- [ ] cosyvoice image 26 GB，重建慢；未來可拆 base image
- [ ] VM 部署是手動 `git pull + compose build`，沒有 CD
