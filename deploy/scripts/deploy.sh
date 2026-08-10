#!/usr/bin/env bash
set -Eeuo pipefail

# Déploiement atomique piloté par une version. Aucun build ni pip install n'est
# exécuté sur le GPU : seules les images préconstruites sont tirées.
PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
VERSION=${1:-${VIDIOAI_VERSION:-}}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}

cd "${PROJECT_DIR}"
test -f "${ENV_FILE}" || { echo "Configuration absente: ${ENV_FILE}" >&2; exit 1; }
test -f "${COMPOSE_FILE}" || { echo "Compose absent: ${COMPOSE_FILE}" >&2; exit 1; }
[[ -n "${VERSION}" && "${VERSION}" != "latest" ]] || { echo "Une version immuable est obligatoire." >&2; exit 1; }
set -a
source "${ENV_FILE}"
set +a
mkdir -p /var/lib/vidioai/backups
cp "${ENV_FILE}" "/var/lib/vidioai/backups/env-$(date +%Y%m%d-%H%M%S)"

# Le backend ne doit jamais démarrer en GPU_PRODUCTION sur les métriques du
# conteneur. Le service natif et son contrat sont donc contrôlés avant le pull.
systemctl is-active --quiet vidioai-host-agent.service \
  || { echo "vidioai-host-agent.service n'est pas actif." >&2; exit 1; }
curl -fsS -H "X-VidioAI-Host-Token: ${HOST_AGENT_TOKEN}" \
  http://127.0.0.1:8091/system | jq -e '.source == "host"' >/dev/null

if [[ -f .current-version ]]; then
  cp .current-version .previous-version
  curl -fsS -X POST -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
    "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}/api/admin/drain" >/dev/null || true
fi

export VIDIOAI_VERSION="${VERSION}"
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" pull
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" up -d --remove-orphans --wait
"${PROJECT_DIR}/deploy/scripts/smoke-test.sh" "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}"
printf '%s\n' "${VERSION}" > .current-version
echo "Déploiement ${VERSION} validé."
