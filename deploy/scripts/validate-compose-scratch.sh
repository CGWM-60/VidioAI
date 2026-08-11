#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
source "${PROJECT_DIR}/deploy/scripts/lib/scratch-storage.sh"

vidioai_require_production_scratch "${ENV_FILE}"
configured=$(VIDIOAI_SCRATCH_DIR="${VIDIOAI_SCRATCH_DIR}" \
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config --format json)

jq -e --arg scratch "${VIDIOAI_SCRATCH_DIR}" '
  (.services.worker.environment.APP_ENV == "GPU_PRODUCTION") and
  (.services.worker.environment.GPU_REQUIRED == "true") and
  (.services.worker.environment.VIDIOAI_MODELS_DIR == "/models") and
  (.services.worker.environment.HF_HOME == "/cache/huggingface") and
  (.services.worker.environment.HF_HUB_CACHE == "/cache/huggingface/hub") and
  (.services.worker.environment.HUGGINGFACE_HUB_CACHE == "/cache/huggingface/hub") and
  (.services.worker.environment.HF_XET_CACHE == "/cache/huggingface/xet") and
  (.services.worker.environment.HF_ASSETS_CACHE == "/cache/huggingface/assets") and
  (.services.worker.environment.TRANSFORMERS_CACHE == "/cache/huggingface/transformers") and
  (.services.worker.environment.DIFFUSERS_CACHE == "/cache/huggingface/diffusers") and
  (.services.worker.environment.TORCH_HOME == "/cache/torch") and
  (.services.worker.environment.XDG_CACHE_HOME == "/cache/xdg") and
  (.services.worker.environment.TMPDIR == "/worker-work") and
  (.services.worker.volumes | any(.source == ($scratch + "/models") and .target == "/models")) and
  (.services.worker.volumes | any(.source == ($scratch + "/cache") and .target == "/cache")) and
  (.services.worker.volumes | any(.source == ($scratch + "/worker-work") and .target == "/worker-work")) and
  (.services.worker.volumes | any(.source == ($scratch + "/work") and .target == "/work")) and
  (.services.backend.volumes | any(.source == ($scratch + "/models") and .target == "/models")) and
  (.services.backend.volumes | any(.source == ($scratch + "/cache") and .target == "/cache")) and
  (.services.backend.volumes | any(.source == ($scratch + "/work") and .target == "/work")) and
  ([.services.worker.volumes[].source, .services.backend.volumes[].source]
    | all(startswith("/var/lib/vidioai/scratch") | not))
' <<<"${configured}" >/dev/null || {
  echo "SCRATCH_COMPOSE_INVALID: les mounts/caches résolus ne ciblent pas exclusivement ${VIDIOAI_SCRATCH_DIR}." >&2
  exit 1
}

echo "SCRATCH_COMPOSE_OK scratch=${VIDIOAI_SCRATCH_DIR}"
