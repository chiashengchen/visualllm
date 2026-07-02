"""GCP Vertex AI training job tools.

trigger_retrain: Submit a custom training job on Vertex AI.
get_job_status:  Poll job status and retrieve metrics when complete.
deploy_model:    Register a trained artifact as a deployable endpoint.

Requires:
  GCP_PROJECT_ID  — GCP project (e.g. visualllm-prod)
  GCP_REGION      — Vertex AI region (e.g. asia-east1)
  google-cloud-aiplatform Python package
"""
from __future__ import annotations

import os
from typing import Any


def _ai_platform():
    try:
        from google.cloud import aiplatform
        return aiplatform
    except ImportError:
        raise ImportError(
            "google-cloud-aiplatform not installed. "
            "Add it to testing_mcp/requirements.txt and reinstall."
        )


def _init_vertex():
    ai = _ai_platform()
    project = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "asia-east1")
    if not project:
        raise ValueError("GCP_PROJECT_ID environment variable not set")
    ai.init(project=project, location=region)
    return ai


async def trigger_retrain(
    model: str,
    dataset_gcs: str,
    config: dict | None = None,
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
) -> dict[str, Any]:
    """Submit a Vertex AI custom training job.

    Args:
        model: Model to train — "stt" (Whisper fine-tune) or "tts" (CosyVoice fine-tune).
        dataset_gcs: GCS URI of the training dataset (e.g. gs://visualllm-data/stt/train/).
        config: Optional training hyperparameters (learning_rate, epochs, etc.).
        machine_type: GCE machine type for the training worker.
        accelerator_type: GPU type (NVIDIA_TESLA_T4 / NVIDIA_TESLA_A100).
        accelerator_count: Number of GPUs.

    Returns:
        Dict with job_id and status.
    """
    if model not in ("stt", "tts"):
        return {"error": f"model must be 'stt' or 'tts', got {model!r}"}

    project = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "asia-east1")
    if not project:
        return {"error": "GCP_PROJECT_ID not set"}

    image = os.getenv(
        "TRAINING_IMAGE_URI",
        f"asia-east1-docker.pkg.dev/{project}/visualllm/training-{model}:latest",
    )

    try:
        ai = _init_vertex()
    except Exception as e:
        return {"error": str(e)}

    job_args = ["--dataset", dataset_gcs, "--model", model]
    if config:
        for k, v in config.items():
            job_args += [f"--{k}", str(v)]

    try:
        job = ai.CustomContainerTrainingJob(
            display_name=f"visualllm-{model}-retrain",
            container_uri=image,
        )
        run = job.run(
            args=job_args,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            sync=False,  # non-blocking; poll with get_job_status
        )
        return {
            "job_id": run.resource_name,
            "display_name": run.display_name,
            "status": "PENDING",
            "dashboard_url": (
                f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
                f"?project={project}"
            ),
        }
    except Exception as e:
        return {"error": f"Vertex AI job submission failed: {e}"}


async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll the status of a Vertex AI training job.

    Args:
        job_id: The resource name returned by trigger_retrain
                (e.g. projects/.../locations/.../customJobs/...).

    Returns:
        Dict with status, and artifact_gcs path when complete.
    """
    try:
        ai = _init_vertex()
    except Exception as e:
        return {"error": str(e)}

    try:
        job = ai.CustomJob.get(job_id)
        state = job.state.name  # PENDING / RUNNING / SUCCEEDED / FAILED / CANCELLED

        result: dict[str, Any] = {
            "job_id": job_id,
            "status": state,
            "create_time": str(job.create_time),
            "update_time": str(job.update_time),
        }

        if state == "SUCCEEDED":
            # The training script should write its artifact to GCS and print the path.
            # Vertex AI surfaces this via job.training_output if the job sets model_id.
            result["artifact_gcs"] = getattr(job, "model_artifact_uri", None)
            result["next_step"] = "Call deploy_model with the artifact_gcs path"
        elif state == "FAILED":
            result["error"] = getattr(job, "error", {})

        return result
    except Exception as e:
        return {"error": f"Could not fetch job {job_id}: {e}"}


async def deploy_model(
    artifact_gcs: str,
    target: str,
) -> dict[str, Any]:
    """Register a trained model artifact and note where to deploy it.

    For now this is a stub that records the artifact location; full automated
    deployment (e.g. updating a GCE VM's model weights) requires target-specific logic.

    Args:
        artifact_gcs: GCS URI of the trained model (e.g. gs://visualllm-data/stt/run-001/).
        target: "stt" (Whisper weights → pipeline STT) or "tts" (CosyVoice → TTS server).

    Returns:
        Dict with instructions for completing the deployment.
    """
    if target not in ("stt", "tts"):
        return {"error": f"target must be 'stt' or 'tts', got {target!r}"}

    steps = {
        "stt": [
            f"Download artifact: gsutil -m cp -r {artifact_gcs} /path/to/whisper-finetuned/",
            "Update STT_MODEL_PATH in .env to point to the new weights",
            "Restart the pipeline: python -m pipeline.main",
        ],
        "tts": [
            f"Download artifact: gsutil -m cp -r {artifact_gcs} /path/to/cosyvoice-finetuned/",
            "Update COSYVOICE_MODEL_DIR in the CosyVoice server config",
            "Restart the CosyVoice vLLM server",
        ],
    }

    return {
        "artifact_gcs": artifact_gcs,
        "target": target,
        "deployment_steps": steps[target],
        "note": "Automated deployment not yet wired; follow the steps above manually.",
    }
