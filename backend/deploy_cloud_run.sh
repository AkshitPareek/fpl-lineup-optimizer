#!/bin/sh
# Deploy the API to Cloud Run. A plan request gets 1 CPU and the service
# scales to zero afterwards. Requires gcloud logged into the target project.
set -eu

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${CLOUD_RUN_REGION:-asia-south1}"
SERVICE="${CLOUD_RUN_SERVICE:-fpl-optimizer-api}"

if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "Set GOOGLE_CLOUD_PROJECT or run: gcloud config set project YOUR_PROJECT" >&2
  exit 1
fi

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --quiet \
  --source "$(CDPATH= cd -- "$(dirname "$0")" && pwd)" \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 2Gi \
  --concurrency 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 2 \
  --cpu-boost \
  --set-env-vars "FRONTEND_URL=https://fploptimize.netlify.app"
