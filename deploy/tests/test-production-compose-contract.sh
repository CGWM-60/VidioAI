#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production.example}
SCRATCH_DIR=/scratch/vidioai

export VIDIOAI_REGISTRY=${VIDIOAI_REGISTRY:-example.invalid/vidioai}
export VIDIOAI_VERSION=${VIDIOAI_VERSION:-contract-test}
export HOST_AGENT_TOKEN=${HOST_AGENT_TOKEN:-host-agent-contract-token-000000000000}
export VIDIOAI_WORKER_TOKEN=${VIDIOAI_WORKER_TOKEN:-worker-contract-token-0000000000000000}
export VIDIOAI_ADMIN_TOKEN=${VIDIOAI_ADMIN_TOKEN:-admin-contract-token-00000000000000000}
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-contract}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-contract-secret}
export AWS_S3_BUCKET=${AWS_S3_BUCKET:-contract-bucket}
export VIDIOAI_SCRATCH_DIR=${SCRATCH_DIR}

VIDIOAI_PROJECT_DIR="${PROJECT_DIR}" \
VIDIOAI_ENV_FILE="${ENV_FILE}" \
VIDIOAI_COMPOSE_FILE="${COMPOSE_FILE}" \
  "${PROJECT_DIR}/deploy/scripts/validate-compose-scratch.sh" >/dev/null

configured=$(VIDIOAI_SCRATCH_DIR="${SCRATCH_DIR}" \
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config --format json)
jq -e '
  (.services.worker.build == null) and
  (.services.backend.build == null) and
  (.services.frontend.build == null)
' <<<"${configured}" >/dev/null

if VIDIOAI_PROJECT_DIR="${PROJECT_DIR}" \
    VIDIOAI_ENV_FILE="${ENV_FILE}" \
    VIDIOAI_COMPOSE_FILE="${PROJECT_DIR}/deploy/tests/fixtures/compose.bad-scratch.yml" \
    "${PROJECT_DIR}/deploy/scripts/validate-compose-scratch.sh" >/dev/null 2>&1; then
  echo "Le mauvais mount /var/lib/vidioai/scratch aurait dû être refusé." >&2
  exit 1
fi

if VIDIOAI_SCRATCH_DIR=/var/lib/vidioai/scratch \
    VIDIOAI_PROJECT_DIR="${PROJECT_DIR}" \
    VIDIOAI_ENV_FILE="${ENV_FILE}" \
    VIDIOAI_COMPOSE_FILE="${COMPOSE_FILE}" \
    "${PROJECT_DIR}/deploy/scripts/validate-compose-scratch.sh" >/dev/null 2>&1; then
  echo "Une variable shell contradictoire aurait dû être refusée." >&2
  exit 1
fi

echo "PRODUCTION_COMPOSE_CONTRACT_OK positive=/scratch/vidioai negative=/var/lib/vidioai/scratch shell_override=refused"
