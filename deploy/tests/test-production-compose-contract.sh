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

configured_with_comfy=$(COMPOSE_PROFILES=comfyui VIDIOAI_SCRATCH_DIR="${SCRATCH_DIR}" \
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config --format json)
jq -e '
  (.services.comfyui.profiles == ["comfyui"]) and
  (.services.comfyui.image | contains("@sha256:")) and
  (.services.comfyui.healthcheck.test | any(contains("http://127.0.0.1:8188/system_stats"))) and
  (.services.comfyui.command | index("--reserve-vram") != null) and
  (.services.comfyui.command | index("12") != null) and
  (.services.comfyui.volumes | any(.target == "/models")) and
  (.services.comfyui.volumes | any(.target == "/opt/comfyui/models")) and
  (.services.comfyui.volumes | any(.target == "/work")) and
  (.services.worker.environment.COMFYUI_URL == "http://comfyui:8188") and
  (.services.worker.environment.VIDIOAI_MODEL_PACKS_DIR == "/opt/vidioai/model-packs") and
  (.services.worker.environment.VIDIOAI_WORKFLOWS_DIR == "/opt/vidioai/workflows") and
  (.services.worker.environment.VIDIOAI_MODEL_PACK_REGISTRY_DIR == "/registry") and
  (.services.backend.environment.VIDIOAI_MODEL_PACK_REGISTRY_DIR == "/registry") and
  (.services.backend.volumes | any(.source == "/var/lib/vidioai/state/model-pack-registry" and .target == "/registry" and ((.read_only // false) == false))) and
  (.services.worker.volumes | any(.source == "/var/lib/vidioai/state/model-pack-registry" and .target == "/registry" and .read_only == true)) and
  (.services.backend.healthcheck.test | index("http://127.0.0.1:8080/api/health") != null) and
  ([.services.backend, .services.worker, .services.frontend]
    | all(.environment.VIDIOAI_VERSION == "contract-test"))
' <<<"${configured_with_comfy}" >/dev/null || { echo 'PRODUCTION_COMPOSE_CONTRACT_DEBUG: production+comfy failed' >&2; jq '.services | {backend,worker,comfyui}' <<<"${configured_with_comfy}" >&2; exit 1; }

local_configured=$(COMPOSE_PROFILES=gpu VIDIOAI_VERSION=contract-test \
  docker compose -f "${PROJECT_DIR}/docker-compose.yml" config --format json)
jq -e --arg project_dir "${PROJECT_DIR}" '
  (.services.backend.build.context == $project_dir) and
  (.services.backend.build.dockerfile == "backend/Dockerfile") and
  (.services.worker.build.context == $project_dir) and
  (.services.worker.build.dockerfile == "worker/Dockerfile") and
  (.services.backend.environment.VIDIOAI_MODEL_PACK_REGISTRY_DIR == "/registry") and
  (.services.worker.environment.VIDIOAI_MODEL_PACK_REGISTRY_DIR == "/registry") and
  ((.services.backend.volumes | map(select(.target == "/registry"))[0].source)
    == (.services.worker.volumes | map(select(.target == "/registry"))[0].source)) and
  (((.services.backend.volumes | map(select(.target == "/registry"))[0].read_only) // false) == false) and
  ((.services.worker.volumes | map(select(.target == "/registry"))[0].read_only) == true) and
  ((.services.backend.volumes | map(select(.target == "/models"))[0].source)
    == (.services.worker.volumes | map(select(.target == "/models"))[0].source))
' <<<"${local_configured}" >/dev/null || { echo 'PRODUCTION_COMPOSE_CONTRACT_DEBUG: local compose failed' >&2; jq '.services | {backend,worker}' <<<"${local_configured}" >&2; exit 1; }

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
