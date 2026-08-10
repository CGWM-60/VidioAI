#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
cd "${PROJECT_DIR}"
test -f .previous-version || { echo "Aucune version précédente connue." >&2; exit 1; }
PREVIOUS_VERSION=$(<.previous-version)
CURRENT_VERSION=$(<.current-version 2>/dev/null || true)
[[ -n "${PREVIOUS_VERSION}" && "${PREVIOUS_VERSION}" != "latest" ]] || { echo "Version de rollback invalide." >&2; exit 1; }
set -a
source .env.production
set +a
curl -fsS -X POST -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
  "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}/api/admin/drain" >/dev/null || true
export VIDIOAI_VERSION="${PREVIOUS_VERSION}"
docker compose -f compose.production.yml --env-file .env.production pull
docker compose -f compose.production.yml --env-file .env.production up -d --remove-orphans --wait
"${PROJECT_DIR}/deploy/scripts/smoke-test.sh" "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}"
printf '%s\n' "${PREVIOUS_VERSION}" > .current-version
if [[ -n "${CURRENT_VERSION}" ]]; then printf '%s\n' "${CURRENT_VERSION}" > .previous-version; fi
echo "Rollback vers ${PREVIOUS_VERSION} terminé."
