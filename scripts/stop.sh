#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_DIR="${PROJECT_ROOT}/.runtime"
PID_FILE="${RUNTIME_DIR}/host-agent.pid"

cd "${PROJECT_ROOT}"
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  # `down` inclut également un éventuel worker démarré avec le profil GPU.
  docker compose --profile gpu down --remove-orphans --timeout 45 >/dev/null
fi
printf 'Frontend        stopped\n'
printf 'Backend         stopped\n'
printf 'Worker          stopped\n'

if [[ -s "${PID_FILE}" ]]; then
  agent_pid=$(<"${PID_FILE}")
  if [[ "${agent_pid}" =~ ^[0-9]+$ ]] && kill -0 "${agent_pid}" 2>/dev/null; then
    kill -TERM "${agent_pid}" 2>/dev/null || true
    for ((attempt=1; attempt<=20; attempt++)); do
      kill -0 "${agent_pid}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "${agent_pid}" 2>/dev/null; then
      kill -KILL "${agent_pid}" 2>/dev/null || true
    fi
  fi
fi
rm -f "${PID_FILE}" "${RUNTIME_DIR}/host-agent.token"
printf 'Host Agent      stopped\n\nVidioAI stopped.\n'

