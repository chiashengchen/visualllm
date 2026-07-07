# 延遲量測 + 硬體用量檢查教學

_針對 GCP 部署（`visualllm-gpu` VM）。所有指令都在你的 Mac 終端機執行，除非特別註明。_

---

## 0. 前置：SSH tunnel（從 Mac 打 CosyVoice 用）

防火牆只開 443 + WebRTC UDP，所以從 Mac 直接打 `34.153.201.22:8001` 是不通的。
要從本機量 TTS，先開 tunnel：

```bash
# 開（-f = 背景執行；之後 Mac 的 localhost:8001 = VM 的 CosyVoice）
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c -- -L 8001:localhost:8001 -N -f

# 確認通了
curl http://localhost:8001/health
# → {"status":"ok", ..., "ready":true}

# 關（找出 tunnel 的 pid 砍掉）
pkill -f "L 8001:localhost:8001"
```

只有你的 Mac 打得到這個 port（tunnel 需要你的 SSH 金鑰，且只綁 127.0.0.1）。

---

## 1. 測延遲

### 方法 A：真實對話的 TTFO（最貼近體感，server 端量測）

Pipeline 內建 `TtfoMeter`，每輪對話自動量「你停止說話 → bot 開始出聲」。
去 `https://34.153.201.22/client/` 講幾句話，然後撈 log：

```bash
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c \
  --command='sudo docker logs pipeline 2>&1 | grep -E "TTFO|TTFB" | tail -20'
```

會看到三種數字：

```
[TTFO OK ] 2.38s (target 8.0s)          ← 端到端（最重要的指標）
OpenAILLMService TTFB: 0.43s            ← LLM 第一個 token
CosyVoiceTTSService TTFB: 1.90s         ← TTS 第一個音訊 chunk
```

斷線時還會印一行 summary（count / median / p95）。

### 方法 B：各 stage 隔離量測（MCP server，找瓶頸用）

`~/testing-mcp-server` 的 `PIPELINE_MODE=real` 會真打 Deepgram / OpenRouter / CosyVoice，
一棒接一棒計時（= sequential 延遲，跟 streaming 的 TTFO 是不同東西，見 §3）。

前置：tunnel 開著（§0），`.env` 已設好（keys + `TTS_BASE_URL=http://localhost:8001`）。

```bash
cd ~/testing-mcp-server
npm run build          # 改過 code 才需要

# 快速 CLI 測一輪（不經 MCP client）：
set -a && source .env && set +a && node --input-type=module -e "
import { config } from './dist/config.js';
import { createProviders } from './dist/pipeline/factory.js';
const { stt, llm, tts } = createProviders(config);
const s = await stt.transcribe({ id:'x', duration_ms:0, simulated_speech_start_ms:0, simulated_speech_end_ms:0 });
console.log('STT :', s.latency_ms.toFixed(0)+'ms →', s.transcript);
const l = await llm.generate(s.transcript || '今天天氣如何？');
console.log('LLM : TTFT', l.ttft_ms.toFixed(0)+'ms, 生成', l.generation_ms.toFixed(0)+'ms');
const t = await tts.synthesize(l.text);
console.log('TTS : TTFA', t.ttfa_ms.toFixed(0)+'ms, 音訊', t.audioDurationMs?.toFixed(0)+'ms');
"
```

要在 Claude Desktop 裡當 MCP tool 用（`run_voice_pipeline` 等），config 加：

```json
{
  "mcpServers": {
    "voice-pipeline-profiler": {
      "command": "node",
      "args": ["/Users/cschen/testing-mcp-server/dist/index.js"],
      "env": {
        "PIPELINE_MODE": "real",
        "DEEPGRAM_API_KEY": "<key>",
        "OPENROUTER_API_KEY": "<key>",
        "TTS_BASE_URL": "http://localhost:8001"
      }
    }
  }
}
```

### 方法 C：純 TTS 壓測（在 VM 上打 localhost，最準）

```bash
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c --command='
time curl -s -o /dev/null -X POST http://localhost:8001/tts/stream \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"今天天氣很好，陽光燦爛，很適合外出活動。\"}"'
```

`total 時間 ÷ 音訊長度 = RTF`（音訊長度 = 回傳 bytes ÷ 48000）。RTF < 1 才追得上播放。

---

## 2. 檢查硬體用量

### 一次性快照

```bash
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c --command='
echo "── GPU ──";  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
echo "── 容器 ──"; sudo docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo "── Host ──"; uptime; free -h | head -2; df -h / | tail -1'
```

### 邊講話邊看（即時監控）

開兩個終端機：一個開著監控，另一個（或瀏覽器）去跟 bot 講話。

```bash
# GPU 每秒刷新 — 講話時看 utilization 跳動
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c \
  --command='watch -n 1 nvidia-smi'

# 或容器 CPU/RAM 即時
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c \
  --command='sudo docker stats'
```

### 抓數據做圖（取樣到 csv）

```bash
gcloud compute ssh visualllm-gpu --zone=asia-northeast1-c --command='
for i in $(seq 1 60); do
  echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  sleep 1
done' > gpu_trace.csv
# 60 秒的 (時間戳, VRAM MiB, GPU%) — 期間去講話，就能看出對話時的曲線
```

### 判讀基準（2026-07-05 實測，詳見 `GCP-DEPLOYMENT-STATUS.md`）

| 指標 | 正常值 | 異常時代表 |
|------|--------|-----------|
| VRAM 閒置 | ~3.3 GiB | 遠高於此 = 有殘留 process，`sudo docker restart cosyvoice` |
| VRAM 合成中峰值 | ~4.8 GiB | 接近 15 GiB = OOM 風險 |
| GPU util 合成中 | 瞬間可到 97%+（正常！） | **持續** 100% 不下來 = 卡住 |
| cosyvoice RAM | ~5.6 GiB | 持續上漲 = memory leak |
| host load1 | < 1.5 | > 4（vCPU 數）= CPU 飽和 |

---

## 3. 讀數字的注意事項

- **TTFO（方法 A）≠ 各 stage 相加（方法 B）。** Pipeline 是 streaming：LLM 第一句一出來就
  送 TTS，三個 stage 重疊執行，所以 TTFO（~2.4-3.8s）遠低於 sequential 加總（~9.6s）。
  方法 B 的用途是**看哪個 stage 慢**，不是預測體感。
- 方法 B 的 STT 是 batch API，數字偏悲觀（live pipeline 用 streaming STT，final transcript 快很多）。
- 方法 B 的 TTS 是整段回覆一次送；live pipeline 逐句送，首句 TTFA 較短。
- 從 Mac 經 tunnel 打 TTS 會多幾百 ms 網路+ssh 開銷；要準就用方法 C（VM 上打 localhost）。
- 改進 response time 時，盯 **TTFO**（開始講得快不快）和**句間空隙**（講起來順不順，
  受 RTF 限制）兩個指標 — 很多優化會顧此失彼。
