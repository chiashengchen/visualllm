#!/usr/bin/env bash
# One-time GCP project setup for VisualLLM.
# Run this once from your local machine with gcloud CLI authenticated.
#
# Usage:
#   chmod +x scripts/gcp_setup.sh
#   ./scripts/gcp_setup.sh visualllm-prod   # pass your desired project ID
#
# After this runs, follow the printed instructions to:
#   1. Add secrets to GCP Secret Manager
#   2. Configure GitHub Actions secrets (WIF_PROVIDER)

set -euo pipefail

PROJECT_ID="${1:-visualllm-prod}"
REGION="asia-east1"
SA_NAME="github-actions"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO_NAME="visualllm"
GITHUB_REPO="${GITHUB_REPO:-cschen/visualllm}"   # override with your org/repo

echo "=== VisualLLM GCP Setup ==="
echo "Project: ${PROJECT_ID}  |  Region: ${REGION}"
echo ""

# ── 1. Create / select project ────────────────────────────────────────────────
if ! gcloud projects describe "${PROJECT_ID}" &>/dev/null; then
  gcloud projects create "${PROJECT_ID}" --name="VisualLLM"
  echo "✓ Created project ${PROJECT_ID}"
else
  echo "✓ Project ${PROJECT_ID} already exists"
fi

gcloud config set project "${PROJECT_ID}"

# Enable billing (must be done manually if using free trial)
echo ""
echo "NOTE: Ensure billing is enabled for ${PROJECT_ID} at:"
echo "  https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}"
echo "  (Skip if you're using the \$300 free trial — billing is auto-enabled)"
echo ""

# ── 2. Enable APIs ────────────────────────────────────────────────────────────
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  iamcredentials.googleapis.com \
  --project="${PROJECT_ID}"
echo "✓ APIs enabled"

# ── 3. Artifact Registry ──────────────────────────────────────────────────────
if ! gcloud artifacts repositories describe "${REPO_NAME}" \
    --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="VisualLLM Docker images"
  echo "✓ Artifact Registry repo created: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
else
  echo "✓ Artifact Registry repo already exists"
fi

# ── 4. Service Account for GitHub Actions ─────────────────────────────────────
if ! gcloud iam service-accounts describe "${SA_EMAIL}" \
    --project="${PROJECT_ID}" &>/dev/null; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="GitHub Actions CI/CD" \
    --project="${PROJECT_ID}"
  echo "✓ Service account created: ${SA_EMAIL}"
else
  echo "✓ Service account already exists"
fi

for ROLE in \
  "roles/run.admin" \
  "roles/artifactregistry.writer" \
  "roles/aiplatform.user" \
  "roles/secretmanager.secretAccessor" \
  "roles/iam.serviceAccountUser"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet
done
echo "✓ IAM roles assigned to ${SA_EMAIL}"

# ── 5. Workload Identity Federation (no long-lived key) ───────────────────────
POOL_ID="github-pool"
PROVIDER_ID="github-provider"

if ! gcloud iam workload-identity-pools describe "${POOL_ID}" \
    --location=global --project="${PROJECT_ID}" &>/dev/null; then
  gcloud iam workload-identity-pools create "${POOL_ID}" \
    --location=global \
    --display-name="GitHub Actions Pool" \
    --project="${PROJECT_ID}"
fi

if ! gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
    --workload-identity-pool="${POOL_ID}" \
    --location=global --project="${PROJECT_ID}" &>/dev/null; then
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
    --workload-identity-pool="${POOL_ID}" \
    --location=global \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --project="${PROJECT_ID}"
fi

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --project="${PROJECT_ID}" \
  --quiet
echo "✓ Workload Identity Federation configured"

# ── 6. Secrets ────────────────────────────────────────────────────────────────
for SECRET in deepgram-key openrouter-key; do
  if ! gcloud secrets describe "${SECRET}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud secrets create "${SECRET}" \
      --replication-policy=automatic \
      --project="${PROJECT_ID}"
    echo "  Created secret: ${SECRET} (add the value with: gcloud secrets versions add ${SECRET} --data-file=-)"
  fi
done
echo "✓ Secret Manager secrets created (values not set — add them manually)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete! Next steps: ==="
echo ""
echo "1. Add secret values:"
echo "   echo -n 'YOUR_DEEPGRAM_KEY'   | gcloud secrets versions add deepgram-key   --data-file=- --project=${PROJECT_ID}"
echo "   echo -n 'YOUR_OPENROUTER_KEY' | gcloud secrets versions add openrouter-key --data-file=- --project=${PROJECT_ID}"
echo ""
echo "2. Add this to your GitHub repo secrets:"
echo "   WIF_PROVIDER = ${WIF_PROVIDER}"
echo "   (Settings → Secrets → Actions → New repository secret)"
echo ""
echo "3. Push to main branch to trigger the first deploy."
echo "   Pipeline will be available at:"
echo "   https://$(gcloud run services describe visualllm-pipeline --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID} 2>/dev/null || echo '<url after first deploy>')/client/"
