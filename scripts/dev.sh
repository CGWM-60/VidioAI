#!/usr/bin/env bash
set -Eeuo pipefail

# `dev.sh` reste au premier plan après le démarrage afin que Ctrl+C arrête tout
# proprement. Définir VIDIOAI_DETACH=1 pour rendre la main en conservant les
# services actifs ; `scripts/stop.sh` les arrêtera ensuite.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_DIR="${PROJECT_ROOT}/.runtime"
PID_FILE="${RUNTIME_DIR}/host-agent.pid"
LOG_FILE="${RUNTIME_DIR}/host-agent.log"
TOKEN_FILE="${RUNTIME_DIR}/host-agent.token"
AGENT_BIN="${PROJECT_ROOT}/host-agent/target/release/vidioai-host-agent"
HTTP_PORT=${VIDIOAI_HTTP_PORT:-3000}
BASE_URL="http://127.0.0.1:${HTTP_PORT}"
STARTUP_COMPLETE=0

say_error() { printf 'Erreur: %s\n' "$*" >&2; }
require_command() {
  command -v "$1" >/dev/null 2>&1 || { say_error "commande requise absente: $1"; exit 1; }
}
wait_for_url() {
  local label=$1 url=$2 token=${3:-} attempts=${4:-60}
  local curl_args=(-fsS --connect-timeout 2 --max-time 5)
  [[ -n "${token}" ]] && curl_args+=(-H "X-VidioAI-Host-Token: ${token}")
  for ((attempt=1; attempt<=attempts; attempt++)); do
    if curl "${curl_args[@]}" "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  say_error "${label} inaccessible après ${attempts}s (${url})"
  return 1
}
cleanup_on_failure() {
  local status=$?
  if [[ ${status} -ne 0 && ${STARTUP_COMPLETE} -eq 0 ]]; then
    say_error "démarrage interrompu; nettoyage des processus déjà lancés"
    "${SCRIPT_DIR}/stop.sh" >/dev/null 2>&1 || true
    [[ -f "${LOG_FILE}" ]] && tail -n 40 "${LOG_FILE}" >&2 || true
  fi
  exit "${status}"
}
stop_on_signal() {
  trap - INT TERM
  printf '\nArrêt demandé…\n'
  "${SCRIPT_DIR}/stop.sh"
  exit 0
}
trap cleanup_on_failure ERR
trap stop_on_signal INT TERM

cd "${PROJECT_ROOT}"
mkdir -p "${RUNTIME_DIR}"
require_command cargo
require_command curl
require_command docker
docker info >/dev/null 2>&1 || { say_error "Docker Desktop/Engine n'est pas démarré"; exit 1; }
docker compose version >/dev/null 2>&1 || { say_error "Docker Compose v2 est absent"; exit 1; }

printf 'VidioAI Development\n\n'

# Ne recompile que lorsque le binaire manque ou qu'une source/configuration est
# plus récente. Le Host Agent est toujours exécuté sur macOS/Linux, jamais dans
# un conteneur.
if [[ ! -x "${AGENT_BIN}" ]] \
  || find "${PROJECT_ROOT}/host-agent/src" "${PROJECT_ROOT}/host-agent/Cargo.toml" \
       -newer "${AGENT_BIN}" -print -quit | grep -q .; then
  printf 'Compilation du Host Agent natif…\n'
  cargo build --manifest-path "${PROJECT_ROOT}/host-agent/Cargo.toml" --release --locked
fi

if [[ ! -s "${TOKEN_FILE}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32 > "${TOKEN_FILE}"
  else
    LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 64 > "${TOKEN_FILE}"
  fi
  chmod 600 "${TOKEN_FILE}"
fi
HOST_AGENT_TOKEN=$(<"${TOKEN_FILE}")
export HOST_AGENT_TOKEN
export HOST_AGENT_URL=${HOST_AGENT_URL:-http://host.docker.internal:8091}
export VIDIOAI_HTTP_PORT="${HTTP_PORT}"

agent_is_running=0
if [[ -s "${PID_FILE}" ]]; then
  existing_pid=$(<"${PID_FILE}")
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    agent_is_running=1
  else
    rm -f "${PID_FILE}"
  fi
fi
if [[ ${agent_is_running} -eq 0 ]]; then
  : > "${LOG_FILE}"
  HOST_AGENT_BIND=0.0.0.0:8091 \
  HOST_AGENT_TOKEN="${HOST_AGENT_TOKEN}" \
    nohup "${AGENT_BIN}" >>"${LOG_FILE}" 2>&1 &
  agent_pid=$!
  printf '%s\n' "${agent_pid}" > "${PID_FILE}"
fi
wait_for_url "Host Agent" "http://127.0.0.1:8091/health" "${HOST_AGENT_TOKEN}" 30
printf 'Host Agent      ✅  natif (PID %s)\n' "$(<"${PID_FILE}")"

# Le worker CUDA est automatiquement activé sur un hôte NVIDIA. En LOCAL sur
# Apple Silicon/CPU, il n'est pas requis par le profil et ne doit pas annoncer
# une fausse disponibilité CUDA.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export COMPOSE_PROFILES=gpu
  export VIDIOAI_WORKER_URL=http://worker:8000
  worker_label="GPU"
else
  unset COMPOSE_PROFILES || true
  export VIDIOAI_WORKER_URL=""
  worker_label="non requis en LOCAL"
fi

BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker compose --progress plain up -d --build --remove-orphans
wait_for_url "Backend" "${BASE_URL}/api/health" "" 120
wait_for_url "Readiness" "${BASE_URL}/api/ready" "" 120
wait_for_url "Frontend" "${BASE_URL}/" "" 120

ready_json=$(curl -fsS "${BASE_URL}/api/ready")
queue_ready=$(printf '%s' "${ready_json}" | grep -q '"queue":true' && printf '✅' || printf '❌')
ws_code=$(curl -sS -o /dev/null --max-time 3 -w '%{http_code}' \
  --http1.1 -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
  -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dmlkaW9haS1kZXYtdGVzdA==' \
  "${BASE_URL}/api/events" 2>/dev/null || true)
[[ "${ws_code}" == "101" ]] || { say_error "WebSocket /api/events non disponible (HTTP ${ws_code})"; exit 1; }

printf 'Backend         ✅\n'
printf 'Frontend        ✅\n'
printf 'Worker          ✅  %s\n' "${worker_label}"
printf 'Database        %s\n' "${queue_ready}"
printf 'WebSocket       ✅\n\n'
printf 'VidioAI ready:\n%s\n' "${BASE_URL}"
printf 'Logs Host Agent: %s\n' "${LOG_FILE}"
STARTUP_COMPLETE=1

if [[ "${VIDIOAI_DETACH:-0}" == "1" ]]; then
  exit 0
fi
printf '\nServices actifs. Ctrl+C pour tout arrêter.\n'
docker compose logs --follow --tail=40
"${SCRIPT_DIR}/stop.sh"
