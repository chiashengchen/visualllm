"""VisualLLM Testing MCP Server.

Exposes tools for evaluating and improving the speech pipeline:
  - run_tts_eval: TTFB + round-trip CER quality check
  - run_stt_eval: WER/CER evaluation against reference transcripts
  - run_e2e_latency: end-to-end pipeline latency measurement
  - trigger_retrain: kick off a Vertex AI training job
  - get_job_status: query training job state

Run locally (for Claude Desktop):
    fastmcp dev testing_mcp/server.py

Run as HTTP server (for remote / Cloud Run):
    fastmcp run testing_mcp/server.py --transport sse --port 8010
"""
from __future__ import annotations

from fastmcp import FastMCP

from testing_mcp.tools.tts_eval import run_tts_eval
from testing_mcp.tools.tts_streaming_eval import run_tts_streaming_eval
from testing_mcp.tools.stt_eval import run_stt_eval
from testing_mcp.tools.e2e_latency import run_e2e_latency
from testing_mcp.tools.gcp_jobs import trigger_retrain, get_job_status, deploy_model

mcp = FastMCP(
    "visualllm-testing",
    instructions=(
        "Tools for evaluating and improving the VisualLLM speech pipeline. "
        "Use run_tts_eval first to check TTS quality (TTFB + round-trip CER on whole "
        "sentences), run_tts_streaming_eval to reproduce the live pipeline's streaming "
        "feed (mock LLM token pacing + first-clause piece splitting — measures first-piece "
        "TTFB and audible seam gaps between pieces that whole-sentence eval can't see), "
        "then run_stt_eval for STT accuracy, and run_e2e_latency for end-to-end timing. "
        "If quality is below threshold, use trigger_retrain to start a GCP training job, "
        "then get_job_status to poll until completion."
    ),
)

mcp.tool()(run_tts_eval)
mcp.tool()(run_tts_streaming_eval)
mcp.tool()(run_stt_eval)
mcp.tool()(run_e2e_latency)
mcp.tool()(trigger_retrain)
mcp.tool()(get_job_status)
mcp.tool()(deploy_model)


if __name__ == "__main__":
    mcp.run()
