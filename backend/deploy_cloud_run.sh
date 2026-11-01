#!/bin/sh
# Deploy the API to Cloud Run. A plan request gets 1 CPU and the service
# scales to zero afterwards. Requires gcloud logged into the target project.
set -eu

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${CLOUD_RUN_REGION:-asia-south1}"
SERVICE="${CLOUD_RUN_SERVICE:-fpl-optimizer-api}"
SOURCE="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"

if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "Set GOOGLE_CLOUD_PROJECT or run: gcloud config set project YOUR_PROJECT" >&2
  exit 1
fi

# /health reports this commit, marked -dirty when deployed with local edits.
if [ -z "${GIT_COMMIT:-}" ]; then
  GIT_COMMIT="$(git -C "$SOURCE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if [ "$GIT_COMMIT" != unknown ] && ! git -C "$SOURCE" diff --quiet HEAD -- . 2>/dev/null; then
    GIT_COMMIT="$GIT_COMMIT-dirty"
  fi
fi
echo "Deploying $SERVICE at commit $GIT_COMMIT to $PROJECT ($REGION)"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --quiet \
  --source "$SOURCE" \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 2Gi \
  --concurrency 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 2 \
  --cpu-boost \
  --set-env-vars "FRONTEND_URL=https://fploptimize.netlify.app,GIT_COMMIT=$GIT_COMMIT"
