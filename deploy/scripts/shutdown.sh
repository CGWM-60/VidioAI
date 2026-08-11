#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
source "${SCRIPT_DIR}/lib/scratch-storage.sh"
cd "${PROJECT_DIR}"
set -a
source "${ENV_FILE}"
set +a
vidioai_require_production_scratch "${ENV_FILE}"

BASE_URL="http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}"
curl -fsS -X POST -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
  "${BASE_URL}/api/admin/drain" >/dev/null
for ((attempt=1; attempt<=${DRAIN_ATTEMPTS:-90}; attempt++)); do
  ACTIVE=$(curl -fsS "${BASE_URL}/api/resources" | jq -r '.queue_active')
  [[ "${ACTIVE}" == "0" ]] && break
  [[ "${attempt}" == "${DRAIN_ATTEMPTS:-90}" ]] && { echo "Drain timeout avec ${ACTIVE} job(s)." >&2; exit 1; }
  sleep 2
done
curl -fsS -X POST -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
  "${BASE_URL}/api/admin/stop" >/dev/null
VIDIOAI_SCRATCH_DIR="${VIDIOAI_SCRATCH_DIR}" \
  docker compose -f compose.production.yml --env-file "${ENV_FILE}" down --timeout 90
if [[ "${VIDIOAI_S3_ENABLED:-false}" == "true" ]]; then
  SNAPSHOT="state/snapshots/$(date -u +%Y%m%dT%H%M%SZ)"
  AWS_ENDPOINT_ARGS=()
  if [[ -n "${AWS_ENDPOINT_URL_S3:-}" ]]; then
    AWS_ENDPOINT_ARGS=(--endpoint-url "${AWS_ENDPOINT_URL_S3}")
  fi
  aws s3 sync "${VIDIOAI_STATE_DIR:-/var/lib/vidioai/state}" \
    "s3://${AWS_S3_BUCKET}/${SNAPSHOT}" \
    --storage-class "${AWS_S3_STORAGE_CLASS:-STANDARD}" \
    "${AWS_ENDPOINT_ARGS[@]}"
fi
if [[ "${VIDIOAI_STOP_HOST_AGENT:-true}" == "true" ]]; then
  systemctl stop vidioai-host-agent.service
fi
echo "VidioAI arrêté après drainage complet."
