# GCP 部署進度紀錄

_更新：2026-07-07。這份文件記錄 GCP VM 部署（**含 MuseTalk avatar**）的現狀。
本機 Windows 部署的狀態仍以 `STATUS.md` 為準。量測方法見 `docs/BENCHMARKING.md`。_

## 現狀：✅ 可用（語音 + 頭像）

`https://34.153.201.22/client/` — STT → LLM → TTS → **MuseTalk talking-head** 端到端正常。
瀏覽器需點過自簽憑證警告（進階 → 繼續前往）。

**⏳ 進行中：vLLM 音質試聽。** 目前 `USE_VLLM=1`（透過 VM 上的
`docker-compose.override.yml`）。T4 時代 vLLM 版音質異常（fp16 + V0），L4 上改跑 bf16
待使用者試聽定案 — 不行就刪 override 檔 + `docker compose up -d cosyvoice` 退回 PyTorch 模式。
已知線索：vLLM 版同句話產出的音訊比 PyTorch 版長（8.6s vs 5.9s），疑似取樣參數差異。

## 版本（git SHA，2026-07-07）

| Repo | Commit | 說明 |
|------|--------|------|
| visualllm | `bef1da3` | docs 更新；功能面最後一筆是 `61b066d`（MUSETALK_FPS 10→12） |
| CosyVoice | `b0d93c2` | 開機 warmup + vLLM stack 烤進 image（USE_VLLM 執行期切換） |

**進度快照**：L4 東京上全套（語音+avatar）可用。vLLM 模式開著（override 檔），
L4 空載 TTS 實測 RTF 0.37（20 字合成 3.1s），但**音質未過關** — vLLM 版音訊比
PyTorch 版長 ~35%（20 字 8.0s vs 5.9s），聽感「不像中文」，根因鎖定為 vLLM 路徑
缺 CosyVoice 原版的 RAS（repetition-aware sampling）→ 語音 token 重複 → 拖音。
PyTorch 模式偶爾也有短句劣化（zero-shot 參考文字比合成文字長的已知警告）。

**下一步（優先序）**：
1. **取樣對齊** — 把 vLLM 路徑的 sampling 對齊原版 RAS（`cosyvoice/llm/llm.py` vllm 分支；
   repetition penalty / top-k / 外層重抽），修好就能吃到 RTF 0.37 又不犧牲音質
2. 短句劣化緩解 — 換短參考音檔（`VOICE_REF`/`VOICE_TEXT`）或 pipeline 端合併短句
3. STT→LLM 提早啟動（interim/partial transcript 就開始生成）— 預估只省 0.2-0.5s，
   因為 streaming STT 的 final 在 VAD 停止後幾乎立即出來；優先度低於 TTS 側
4. L4 乾淨基準（音質定案後）

## 部署架構

- **VM**: `visualllm-gpu`，zone **`asia-northeast1-c`（東京）**，外部 IP `34.153.201.22`
  - `g2-standard-4`（4 vCPU / 16GB RAM）+ **NVIDIA L4 24GB**（Ada）
  - 磁碟 160GB（2026-07-07 從 100GB 擴容 — vLLM image 需要空間）
  - 2026-07-07 從首爾 T4（`asia-northeast3-c`）用 machine image 整機遷移。
    台灣 `asia-east1` 的 L4 三個 zone 都 STOCKOUT，東京是最近的有貨區。
    舊快照 `visualllm-t4-snapshot`（62GB）還在，確認穩定後可刪（~$3/月）。
- **docker compose 四個服務**：
  - `cosyvoice` — CosyVoice**2**-0.5B TTS，GPU，:8001。image 已含 vLLM stack
    （vllm 0.9.2 / torch 2.7 / transformers 4.51.3 / numpy2-ABI 修復），
    `USE_VLLM` 環境變數決定推理引擎。開機自動 warmup（燒掉 cuDNN autotune 的
    首輪 ~10s TTFB）。CosyVoice 需要 prompt_embeds → vLLM 永遠退 V0 引擎（跟 GPU 無關）。
  - `musetalk` — MuseTalk v1.5 avatar，GPU，:8002。權重（~5GB）首次啟動自動下載進
    volume，頭像 `assets/avatar.jpg`（**不進 git**，`gcloud compute scp` 上去，掛載唯讀）。
    `MUSETALK_FPS=12`（跟 pipeline 兩處必須一致）。
  - `pipeline` — Pipecat，**`network_mode: host`**（WebRTC UDP 必須直綁 host IP），
    `ENABLE_AVATAR=1`，:7860
  - `nginx` — HTTPS 自簽憑證，443 → `host.docker.internal:7860`
- **STUN**: `pipeline/main.py::_configure_stun_servers()` 注入 google STUN（GCP 1:1 NAT 必需）
- **防火牆**（tag `visualllm`）: 只開 tcp 443 + udp 49152-65535（WebRTC media）。
  benchmark TTS 要先 ssh tunnel（見 `docs/BENCHMARKING.md` §0）
- 設定: `LANGUAGE=zh`、LLM=openrouter（gemini-2.5-flash-lite）、STT=Deepgram nova-2

## 延遲實測

**L4 乾淨數據還沒量**（上次手動測試時 vLLM image build 正在搶 CPU，數字被灌水：
TTFO median 6.76s）。vLLM 音質定案後要重測一輪，下面是歷史對照：

| 指標 | T4 首爾（07-05，PyTorch，無 avatar） | T4（07-07，含 avatar） | 5060 Ti 本機（文件 `docs/measure_data.js`，en，vLLM） |
|------|------|------|------|
| TTFO | 2.4–3.8s | 6.3–7.6s | **3.69s** |
| LLM TTFB | 0.47–0.69s | 0.66–1.33s | 0.68s |
| TTS 首 chunk | 1.4–3.2s | 3.9–5.8s | 1.75s（zh ~2.3s） |
| Avatar fps | — | 7.4–8.4（設 10） | 12.2（設 12，跑滿） |

**TTS 引擎對比（20 字中文句，總合成時間）**：
- L4 PyTorch（torch 2.7）：7.0s（RTF ~1.19）
- L4 vLLM（bf16, V0）：5.6s 但音訊變長 8.6s（**RTF 0.65**）
- 關鍵發現：CosyVoice 的自回歸生成是**延遲綁定**（GPU util 只有 ~22%），
  所以 T4→L4 純算力提升對 PyTorch 模式幾乎沒幫助；vLLM 的調度優化才是快的來源。

## 已知問題 / 待辦

- [ ] **vLLM 音質試聽**（進行中，見上）；音訊變長的疑點要查 vllm 路徑的 sampling 參數
- [ ] **L4 乾淨基準測試**（vLLM 定案後跑一輪：TTFO / TTS bench / avatar fps@12 / 資源用量）
- [ ] fps 12 長回覆的 end drift 觀察（T4 上 render 跟不上 12 時 drift 會放大；L4 待驗證）
- [ ] **Disconnect 後立刻 Connect 會卡住** — 瀏覽器端問題（playground 沒送 offer），
  非 server 端。Workaround：重新整理頁面（F5）再連，或等幾秒再按一次 Connect
- [ ] 刪 `visualllm-t4-snapshot` machine image（確認 L4 穩定後）
- [ ] 自簽憑證：每個新瀏覽器都要手動信任。可改 Let's Encrypt（需要網域）
- [ ] VM 部署是手動 `git pull + compose build`，沒有 CD
- [ ] 候選升級：CosyVoice 3（`HF_MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，
  server 支援切換，但推理相容性 + vLLM 相容性未驗證）

## 歷史（已解決）

- WebRTC 連不上 → HTTPS 必需（nginx 自簽）+ STUN + host network（Docker NAT 擋 UDP）
- CosyVoice `TypeError: Invalid file: tensor` → `inference_zero_shot` 要傳 wav **路徑**
- T4 上 vLLM 相依地獄 → torch 2.7 / numpy 2 ABI（onnxruntime、pyarrow 升版 + pyworld 重編）、
  transformers 釘 4.51.3、`ModelRegistry.register_model()` 手動註冊、
  `COSYVOICE_VLLM_GPU_UTIL`（寫死 0.2 會 crash）
- MuseTalk Docker 化 → 逐檔權重檢查（gdown≥5 語法變了）、`.dockerignore` 解封、
  pipeline Dockerfile 補 COPY `musetalk_video.py`
- Cloud Run CD 移除（從未成功部署過 — GitHub secrets 根本沒設）；GPU 配額 all-regions=1
  是 T4→L4 只能刪機重開（而非並行遷移）的原因
