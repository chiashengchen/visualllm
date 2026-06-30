// Front-end for the local CosyVoice TTS demo.
// Talks to the SAME-ORIGIN existing endpoints: GET /health and POST /tts.

const $ = (id) => document.getElementById(id);
const textEl = $("text");
const btn = $("generateBtn");
const btnLabel = btn.querySelector(".btn-label");
const spinner = btn.querySelector(".spinner");
const statusEl = $("status");
const playerEl = $("player");
const audioEl = $("audio");
const metaEl = $("meta");
const downloadEl = $("download");
const speedEl = $("speed");
const speedVal = $("speedVal");
const badge = $("engineBadge");

let lastUrl = null; // revoke previous object URLs to avoid leaks

function setStatus(msg, kind = "info") {
  statusEl.textContent = msg;
  statusEl.className = `status ${kind}`;
}

function setLoading(on) {
  btn.disabled = on;
  spinner.hidden = !on;
  btnLabel.textContent = on ? "Generating…" : "Generate Speech";
}

// Live speed label
speedEl.addEventListener("input", () => {
  speedVal.textContent = `${parseFloat(speedEl.value).toFixed(1)}×`;
});

// Sample chips
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textEl.value = chip.dataset.sample;
    textEl.focus();
  });
});

// Check engine health on load
async function checkHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    if (r.ok && j.status === "ok") {
      badge.textContent = `engine ready · ${j.device} · ${j.sample_rate} Hz`;
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = "engine loading…";
      badge.className = "badge badge-muted";
      setTimeout(checkHealth, 2000);
    }
  } catch (e) {
    badge.textContent = "engine offline";
    badge.className = "badge badge-err";
  }
}

async function generate() {
  const text = textEl.value.trim();
  if (!text) {
    setStatus("Please enter some text first.", "err");
    textEl.focus();
    return;
  }

  setLoading(true);
  setStatus("Synthesizing on the local CosyVoice engine… (CPU ≈ 2× audio length)", "info");

  const started = performance.now();
  try {
    const resp = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, speed: parseFloat(speedEl.value) }),
    });

    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const err = await resp.json();
        if (err.detail) detail = typeof err.detail === "string"
          ? err.detail : JSON.stringify(err.detail);
      } catch (_) {}
      throw new Error(detail);
    }

    const blob = await resp.blob();
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    lastUrl = URL.createObjectURL(blob);

    audioEl.src = lastUrl;
    downloadEl.href = lastUrl;
    playerEl.hidden = false;

    const gen = resp.headers.get("X-Generation-Seconds");
    const dur = resp.headers.get("X-Audio-Seconds");
    const rtf = resp.headers.get("X-RTF");
    const wall = ((performance.now() - started) / 1000).toFixed(2);
    metaEl.textContent =
      `audio ${dur ?? "?"}s · generated in ${gen ?? wall}s · RTF ${rtf ?? "?"}`;

    setStatus("✓ Audio generated. Playing…", "ok");
    try { await audioEl.play(); } catch (_) { /* autoplay may be blocked; user can press play */ }
  } catch (e) {
    playerEl.hidden = true;
    setStatus(`✗ Generation failed: ${e.message}`, "err");
  } finally {
    setLoading(false);
  }
}

btn.addEventListener("click", generate);
// Ctrl/Cmd+Enter to generate
textEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
});

checkHealth();
