#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
WAIT_TIMEOUT=${VIDIOAI_DEPLOY_WAIT_TIMEOUT:-180}
WAIT_INTERVAL=${VIDIOAI_DEPLOY_WAIT_INTERVAL:-2}

compose() {
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" "$@"
}

has_service() {
  local service=${1:?service requis}
  compose config --services | grep -Fxq "${service}"
}

service_container_id() {
  local service=${1:?service requis}
  compose ps -q "${service}" 2>/dev/null || true
}

service_state() {
  local service=${1:?service requis}
  local cid
  cid=$(service_container_id "${service}")
  if [[ -z "${cid}" ]]; then
    printf 'absent|none|none\n'
    return 0
  fi
  local status health
  status=$(docker inspect -f '{{.State.Status}}' "${cid}" 2>/dev/null || echo "unknown")
  health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${cid}" 2>/dev/null || echo "none")
  printf '%s|%s|%s\n' "${status}" "${health}" "${cid}"
}

wait_for_service() {
  local service=${1:?service requis}
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < deadline )); do
    IFS='|' read -r status health _cid <<<"$(service_state "${service}")"
    echo "[wait] ${service}: status=${status} health=${health}"
    if [[ "${status}" == "running" && ( "${health}" == "healthy" || "${health}" == "none" ) ]]; then
      return 0
    fi
    sleep "${WAIT_INTERVAL}"
  done
  echo "Service ${service} non prêt après ${WAIT_TIMEOUT}s." >&2
  return 1
}

dump_diagnostics() {
  echo "=== Diagnostic rollback ===" >&2
  compose ps -a >&2 || true
  for service in worker backend frontend proxy; do
    compose logs --tail=100 "${service}" >&2 || true
  done
}

trap dump_diagnostics ERR

cd "${PROJECT_DIR}"
test -f .previous-version || { echo "Aucune version précédente connue." >&2; exit 1; }
PREVIOUS_VERSION=$(<.previous-version)
CURRENT_VERSION=$(cat .current-version 2>/dev/null || true)
[[ -n "${PREVIOUS_VERSION}" && "${PREVIOUS_VERSION}" != "latest" ]] || { echo "Version de rollback invalide." >&2; exit 1; }
set -a
source "${ENV_FILE}"
set +a
curl -fsS -X POST -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
  "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}/api/admin/drain" >/dev/null || true
export VIDIOAI_VERSION="${PREVIOUS_VERSION}"
compose pull
for service in worker backend frontend proxy; do
  if has_service "${service}"; then
    compose up -d --remove-orphans "${service}"
    wait_for_service "${service}"
  fi
done
"${PROJECT_DIR}/deploy/scripts/smoke-test.sh" "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}"
printf '%s\n' "${PREVIOUS_VERSION}" > .current-version
if [[ -n "${CURRENT_VERSION}" ]]; then printf '%s\n' "${CURRENT_VERSION}" > .previous-version; fi
echo "Rollback vers ${PREVIOUS_VERSION} terminé."
