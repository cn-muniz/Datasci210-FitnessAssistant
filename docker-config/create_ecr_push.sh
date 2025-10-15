#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
WEB_REPO="${WEB_REPO:-fitness-web}"
API_REPO="${API_REPO:-recipe-api}"
PROJECT_NAME="${PROJECT_NAME:-fitnessapp}"

ensure_repo() {
  local repo_name=$1
  if ! aws ecr describe-repositories --repository-name "${repo_name}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    aws ecr create-repository --repository-name "${repo_name}" --region "${AWS_REGION}" >/dev/null
  fi
}

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
WEB_IMAGE="${REGISTRY}/${WEB_REPO}:${IMAGE_TAG}"
API_IMAGE="${REGISTRY}/${API_REPO}:${IMAGE_TAG}"

ensure_repo "${WEB_REPO}"
ensure_repo "${API_REPO}"

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

docker compose -f "${SCRIPT_DIR}/docker-compose.yml" -p "${PROJECT_NAME}" build web api

LOCAL_WEB_IMAGE="${PROJECT_NAME}-web:latest"
LOCAL_API_IMAGE="${PROJECT_NAME}-api:latest"

docker tag "${LOCAL_WEB_IMAGE}" "${WEB_IMAGE}"
docker tag "${LOCAL_API_IMAGE}" "${API_IMAGE}"

docker push "${WEB_IMAGE}"
docker push "${API_IMAGE}"
