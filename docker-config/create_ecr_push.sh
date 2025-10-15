#!/usr/bin/env bash
set -euo pipefail

# -------- Locators (make paths absolute) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WEB_CONTEXT="${REPO_ROOT}"
WEB_DOCKERFILE="${REPO_ROOT}/docker-config/Dockerfile"

API_CONTEXT="${REPO_ROOT}/VectorDB/recipes-vdb/app"
API_DOCKERFILE="${API_CONTEXT}/Dockerfile"

[[ -f "$WEB_DOCKERFILE" ]] || { echo "Missing WEB Dockerfile at $WEB_DOCKERFILE"; exit 1; }
[[ -f "$API_DOCKERFILE" ]] || { echo "Missing API Dockerfile at $API_DOCKERFILE"; exit 1; }

# -------- Config --------
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
WEB_REPO="${WEB_REPO:-fitness-web}"
API_REPO="${API_REPO:-recipe-api}"
PLATFORMS="${PLATFORMS:-linux/amd64}"   # use "linux/amd64,linux/arm64" for multi-arch

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
WEB_IMAGE="${REGISTRY}/${WEB_REPO}:${IMAGE_TAG}"
API_IMAGE="${REGISTRY}/${API_REPO}:${IMAGE_TAG}"

ensure_repo() {
  local repo="$1"
  aws ecr describe-repositories --repository-name "$repo" --region "$AWS_REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION" >/dev/null
}

echo "Ensuring ECR repos exist…"
ensure_repo "$WEB_REPO"
ensure_repo "$API_REPO"

echo "Logging in to ECR…"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"

# Ensure buildx is available
docker buildx create --use >/dev/null 2>&1 || true

# pick a stable cache tag per repo
WEB_CACHE="${REGISTRY}/${WEB_REPO}:buildcache"
API_CACHE="${REGISTRY}/${API_REPO}:buildcache"

docker buildx build \
  --platform "$PLATFORMS" \
  --file "$WEB_DOCKERFILE" \
  --tag "$WEB_IMAGE" \
  --cache-from "type=registry,ref=${WEB_CACHE}" \
  --cache-to   "type=registry,ref=${WEB_CACHE},mode=max" \
  --push \
  "$WEB_CONTEXT"

docker buildx build \
  --platform "$PLATFORMS" \
  --file "$API_DOCKERFILE" \
  --tag "$API_IMAGE" \
  --cache-from "type=registry,ref=${API_CACHE}" \
  --cache-to   "type=registry,ref=${API_CACHE},mode=max" \
  --push \
  "$API_CONTEXT"

echo "Done."
echo " WEB: $WEB_IMAGE"
echo " API: $API_IMAGE"
