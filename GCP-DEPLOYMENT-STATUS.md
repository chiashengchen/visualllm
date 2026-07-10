# GCP 部署進度紀錄

_更新：2026-07-10。這份文件記錄 GCP VM 部署（**含 MuseTalk avatar**）的現狀。
本機 Windows 部署的狀態仍以 `STATUS.md` 為準。量測方法見 `docs/BENCHMARKING.md`。_

## 現狀：✅ 全面可用（語音 + 頭像，音質已驗收）

`https://34.153.201.22/client/`（prebuilt UI）/ `https://34.153.201.22/nimbus/`（自訂 UI）
瀏覽器需點過自簽憑證警告。使用者 2026-07-10 實測通過。

**現行組態**：L4 + **vLLM**（RAS 取樣修復）+ **OpenCC 繁→簡**（T2S）+ 美佳參考音色。
品質：整句 CER 0.06 / streaming 路徑 CER 0.0；TTS TTFB ~2.3s（比 PyTorch 快 44%）、
RTF ~0.4；live TTFO median 3.39s。

## 版本（git SHA，2026-07-10）

| Repo | Commit | 說明 |
|------|--------|------|
| visualllm | `3ca253e` | testing_mcp streaming eval + docs |
| CosyVoice | `82356c4` | **T2S 根因修復**；含 RAS（ab52574）、warmup、vLLM stack |

## zh 音質破案史（2026-07-10 收案）

**根因 = CosyVoice 文字前端吃繁體中文會崩**：繁體長句 ~10 字後劣化成亂碼
（簡體完美），pipeline 是 zh-TW 所以從第一天（T4）就爛。A/B 矩陣排除了
vLLM / RAS / fp16 / streaming flow / 相依版本（oldstack 重建對照）/ 參考音檔全部嫌疑，
最後用「簡體 vs 繁體同句對照」鎖定。**修復 = server 端 OpenCC 繁→簡**
（`COSYVOICE_T2S=0` 可關），唸出來的普通話一模一樣。CER 0.45 → **0.09**。
之前「vLLM 音質怪」的觀察其實是這個 bug 的誤判 — vLLM 平反後重新上線。

評測工具 = `testing_mcp` 的 `run_tts_eval`（整句）+ `run_tts_streaming_eval`
（mock LLM token 節奏 + first-piece 切片，量接縫空隙）。注意 Deepgram 評審的
雜訊底線 CER ~0.1、數字會正規化（三點→3）造成假錯誤。

## 部署架構

- **VM**: `visualllm-gpu`，zone **`asia-northeast1-c`（東京）**，外部 IP `34.153.201.22`
  - `g2-standard-4`（4 vCPU / 16GB RAM）+ **NVIDIA L4 24GB**（Ada），磁碟 160GB
  - 2026-07-07 從首爾 T4 用 machine image 遷移（台灣 L4 全 STOCKOUT；
    GPU 配額 all-regions=1 所以只能刪機重開）。舊快照 `visualllm-t4-snapshot` 可刪（~$3/月）
- **docker compose 四個服務**：
  - `cosyvoice` — CosyVoice**2**-0.5B，GPU，:8001。`USE_VLLM` 執行期切換引擎
    （vLLM 因 prompt_embeds 永遠退 V0，跟 GPU 無關）；開機 warmup；T2S 內建。
    **現行 override 檔**（VM 上，不進 git）：`USE_VLLM=1`、`COSYVOICE_VLLM_GPU_UTIL=0.3`、
    美佳參考音色（`VOICE_REF_WAV`/`VOICE_PROMPT_TEXT`，wav 在 `assets/meijia_ref.wav`，
    macOS `say -v Meijia` 生成）。刪 override + `compose up -d cosyvoice` = 回 PyTorch/預設音色
  - `musetalk` — MuseTalk v1.5 avatar，GPU，:8002。權重自動下載進 volume；
    頭像 `assets/avatar.jpg`（不進 git，scp 上 VM）。`MUSETALK_FPS=12`（兩處必須一致）
  - `pipeline` — Pipecat，`network_mode: host`（WebRTC UDP 必需），`ENABLE_AVATAR=1`，:7860
  - `nginx` — HTTPS 自簽，443 → `host.docker.internal:7860`
- **STUN**: `_configure_stun_servers()` 注入 google STUN（GCP 1:1 NAT 必需）
- **防火牆**（tag `visualllm`）: 只開 tcp 443 + udp 49152-65535；benchmark 走 ssh tunnel
- 設定: `LANGUAGE=zh`、LLM=openrouter（gemini-2.5-flash-lite）、STT=Deepgram nova-2

## 延遲實測（最終，2026-07-10，vLLM+T2S）

| 指標 | T4 首爾（07-05） | **L4 東京（現在）** | 5060 Ti 本機（en） |
|------|------|------|------|
| TTFO | 6.3–7.6s（含 avatar） | **2.99–3.39s median**（一次 6.2s outlier） | 3.69s |
| LLM TTFB | 0.66–1.33s | 0.4–0.5s | 0.68s |
| TTS 首 chunk | 3.9–5.8s | **~2.3s** | 1.75s（en） |
| TTS RTF | ~1.0 | **~0.4** | — |
| Avatar fps | 7.4–8.4（設 10 跑不滿） | **12.2（設 12 跑滿）** | 12.2 |
| 整句 CER | （壞的，當時沒量） | **0.06–0.09** | — |

關鍵發現：CosyVoice 自回歸生成是**延遲綁定**（GPU util ~22%），換強 GPU 對 PyTorch
模式沒幫助，vLLM 的調度優化才是快的來源。

## 已知問題 / 待辦

- [ ] 冷啟動後第一次合成偶發靜音抖動（重跑即過；可把 warmup 改成長句多跑一次）
- [ ] streaming 曾出現一次 +0.54s 接縫（`run_tts_streaming_eval` 可持續監測）
- [ ] Disconnect 後立刻 Connect 卡住 — 瀏覽器端（不送 offer）；F5 或再按一次
- [ ] 刪 `visualllm-t4-snapshot` machine image
- [ ] 自簽憑證 → Let's Encrypt（需要網域）
- [ ] VM 部署是手動 `git pull + compose build`，沒有 CD
- [ ] 美佳參考音色要不要轉正（寫進 compose/image）或換自選音色
- [ ] 候選升級：CosyVoice 3（`Fun-CosyVoice3-0.5B-2512`，相容性未驗證）

## 歷史（已解決）

- **zh 音質**（見上方破案史）— T2S 修復，vLLM 平反
- OpenRouter key 失效（401 User not found，2026-07-10）→ 換新 key
- WebRTC 連不上 → HTTPS（nginx 自簽）+ STUN + host network（Docker NAT 擋 UDP）
- CosyVoice `TypeError: Invalid file: tensor` → `inference_zero_shot` 要傳 wav **路徑**
- vLLM 相依地獄 → torch 2.7 / numpy 2 ABI（onnxruntime、pyarrow 升版 + pyworld 重編）、
  transformers 釘 4.51.3、`ModelRegistry.register_model()`、`COSYVOICE_VLLM_GPU_UTIL`
- Dockerfile.tts 層序（依賴在前、`COPY . .` 最後）→ 改 code 秒級重建
- MuseTalk Docker 化 → 權重逐檔檢查（gdown≥5 語法）、`.dockerignore` 解封、
  pipeline 補 COPY `musetalk_video.py`
- Cloud Run CD 移除（從未成功部署過）
