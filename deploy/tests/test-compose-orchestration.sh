#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/docker-compose.yml}
HTTP_PORT=${VIDIOAI_HTTP_PORT:-18080}
COMPOSE_PROJECT=${VIDIOAI_COMPOSE_PROJECT:-vidioai-ci}

compose() {
  VIDIOAI_HTTP_PORT="${HTTP_PORT}" docker compose -f "${COMPOSE_FILE}" -p "${COMPOSE_PROJECT}" "$@"
}

cleanup() {
  compose down -v --remove-orphans || true
}

on_error() {
  compose ps -a || true
  compose logs --tail=160 backend frontend proxy || true
}

trap on_error ERR
trap cleanup EXIT

wait_http() {
  local url=${1:?url requis}
  local attempts=${2:-90}
  local delay=${3:-2}
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}

wait_http_status() {
  local url=${1:?url requis}
  local expected=${2:?status attendu requis}
  local attempts=${3:-90}
  local delay=${4:-2}
  local status
  for ((i=1; i<=attempts; i++)); do
    status=$(curl -sS -o /dev/null -w '%{http_code}' "${url}" || echo "000")
    if [[ "${status}" == "${expected}" ]]; then
      return 0
    fi
    sleep "${delay}"
  done
  echo "Statut HTTP inattendu pour ${url}: ${status} (attendu ${expected})" >&2
  return 1
}

assert_proxy_routes_ok() {
  wait_http_status "http://127.0.0.1:${HTTP_PORT}/api/health" "200" || {
    compose ps -a
    compose logs --tail=160 proxy backend
    exit 1
  }
  wait_http_status "http://127.0.0.1:${HTTP_PORT}/" "200" || {
    compose ps -a
    compose logs --tail=160 proxy frontend
    exit 1
  }
  wait_http_status "http://127.0.0.1:${HTTP_PORT}/models" "200" || {
    compose ps -a
    compose logs --tail=160 proxy frontend
    exit 1
  }
}

container_ip() {
  local service=${1:?service requis}
  local cid
  cid=$(compose ps -q "${service}" || true)
  [[ -n "${cid}" ]] || return 1
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${cid}" 2>/dev/null || true
}

wait_service_running() {
  local service=${1:?service requis}
  local attempts=${2:-90}
  local delay=${3:-2}
  for ((i=1; i<=attempts; i++)); do
    cid=$(compose ps -q "${service}" || true)
    if [[ -n "${cid}" ]]; then
      status=$(docker inspect -f '{{.State.Status}}' "${cid}" 2>/dev/null || echo "unknown")
      health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${cid}" 2>/dev/null || echo "none")
      if [[ "${status}" == "running" && ( "${health}" == "healthy" || "${health}" == "none" ) ]]; then
        return 0
      fi
    fi
    sleep "${delay}"
  done
  return 1
}

cd "${PROJECT_DIR}"

compose up -d --build

for svc in backend frontend proxy; do
  wait_service_running "${svc}" || { compose ps -a; compose logs --tail=120 "${svc}"; exit 1; }
done

assert_proxy_routes_ok

backend_before=$(compose ps -q backend)
frontend_before=$(compose ps -q frontend)
backend_ip_before=$(container_ip backend)
frontend_ip_before=$(container_ip frontend)

for round in 1 2 3; do
  echo "[round ${round}] Recréation backend sans restart proxy"
  compose up -d --force-recreate backend
  wait_service_running backend || { compose ps -a; compose logs --tail=160 backend proxy; exit 1; }
  backend_after=$(compose ps -q backend)
  [[ "${backend_before}" != "${backend_after}" ]] || { echo "Backend non recréé (round ${round})." >&2; exit 1; }
  backend_ip_after=$(container_ip backend)
  if [[ -n "${backend_ip_before}" && -n "${backend_ip_after}" && "${backend_ip_before}" != "${backend_ip_after}" ]]; then
    echo "Backend IP modifiée: ${backend_ip_before} -> ${backend_ip_after}"
  else
    echo "Backend recréé sans changement d'IP observable (pool Docker inchangé)."
  fi
  assert_proxy_routes_ok
  backend_before="${backend_after}"
  backend_ip_before="${backend_ip_after}"

  echo "[round ${round}] Recréation frontend sans restart proxy"
  compose up -d --force-recreate frontend
  wait_service_running frontend || { compose ps -a; compose logs --tail=160 frontend proxy; exit 1; }
  frontend_after=$(compose ps -q frontend)
  [[ "${frontend_before}" != "${frontend_after}" ]] || { echo "Frontend non recréé (round ${round})." >&2; exit 1; }
  frontend_ip_after=$(container_ip frontend)
  if [[ -n "${frontend_ip_before}" && -n "${frontend_ip_after}" && "${frontend_ip_before}" != "${frontend_ip_after}" ]]; then
    echo "Frontend IP modifiée: ${frontend_ip_before} -> ${frontend_ip_after}"
  else
    echo "Frontend recréé sans changement d'IP observable (pool Docker inchangé)."
  fi
  assert_proxy_routes_ok
  frontend_before="${frontend_after}"
  frontend_ip_before="${frontend_ip_after}"
done

wait_service_running proxy || { compose ps -a; compose logs --tail=160 proxy; exit 1; }

if compose ps -a --format json \
  | jq -s -e '
      map(if type == "array" then .[] else . end)
      | map(select((.State // "" | ascii_downcase) | IN("created", "exited", "restarting")))
      | length > 0
    ' >/dev/null; then
  compose ps -a
  exit 1
fi

echo "Compose orchestration test: OK"
