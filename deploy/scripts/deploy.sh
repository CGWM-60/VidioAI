#!/usr/bin/env bash
set -Eeuo pipefail

# Déploiement atomique piloté par une version. Aucun build ni pip install n'est
# exécuté sur le GPU : seules les images préconstruites sont tirées.
PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VERSION=${1:-${VIDIOAI_VERSION:-}}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
WAIT_TIMEOUT=${VIDIOAI_DEPLOY_WAIT_TIMEOUT:-180}
WAIT_INTERVAL=${VIDIOAI_DEPLOY_WAIT_INTERVAL:-2}
BACKUP_DIR=${VIDIOAI_BACKUP_DIR:-/var/lib/vidioai/backups}
source "${SCRIPT_DIR}/lib/scratch-storage.sh"

compose() {
  VIDIOAI_SCRATCH_DIR="${VIDIOAI_SCRATCH_DIR:?Scratch non configuré}" \
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

dump_diagnostics() {
  echo "=== Diagnostic Compose ===" >&2
  compose ps -a >&2 || true
  for service in comfyui worker backend frontend proxy; do
    if ! has_service "${service}"; then
      continue
    fi
    IFS='|' read -r status health cid <<<"$(service_state "${service}")"
    if [[ "${status}" != "running" || ( "${health}" != "healthy" && "${health}" != "none" ) ]]; then
      echo "--- ${service}: status=${status} health=${health} ---" >&2
      compose logs --tail=100 "${service}" >&2 || true
      if [[ -n "${cid}" && "${cid}" != "none" ]]; then
        docker inspect "${cid}" --format '{{json .State}}' >&2 || true
      fi
    fi
  done
}

wait_for_service() {
  local service=${1:?service requis}
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < deadline )); do
    IFS='|' read -r status health _cid <<<"$(service_state "${service}")"
    echo "[wait] ${service}: status=${status} health=${health}"
    case "${status}" in
      created|exited|dead|removing)
        echo "Service ${service} bloqué dans un état terminal/non-démarré: ${status}" >&2
        return 1
        ;;
    esac
    if [[ "${status}" == "running" && ( "${health}" == "healthy" || "${health}" == "none" ) ]]; then
      return 0
    fi
    sleep "${WAIT_INTERVAL}"
  done
  echo "Service ${service} non prêt après ${WAIT_TIMEOUT}s." >&2
  return 1
}

verify_stack_healthy() {
  local failed=0
  for service in comfyui worker backend frontend proxy; do
    if ! has_service "${service}"; then
      continue
    fi
    IFS='|' read -r status health _cid <<<"$(service_state "${service}")"
    if [[ "${status}" != "running" || ( "${health}" != "healthy" && "${health}" != "none" ) ]]; then
      echo "Service invalide: ${service} status=${status} health=${health}" >&2
      failed=1
    fi
  done
  return "${failed}"
}

check_proxy_route() {
  local path=${1:?path requis}
  local expected_status=${2:-200}
  local url="http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}${path}"
  local status

  # Chemin nominal: requête HTTP réussie (inclut les environnements de tests
  # mockés qui ne renvoient pas forcément un code numérique via -w).
  if curl -fsS "${url}" >/dev/null; then
    return 0
  fi

  status=$(curl -sS -o /dev/null -w '%{http_code}' "${url}" || echo "000")
  if [[ "${status}" == "${expected_status}" ]]; then
    return 0
  fi

  echo "Vérification proxy en échec sur ${path}: HTTP ${status} (attendu ${expected_status})." >&2
  if [[ "${status}" == "502" ]]; then
    echo "Nginx renvoie 502: collecte automatique des logs proxy/backend/frontend." >&2
    compose logs --tail=200 proxy backend frontend >&2 || true
  fi
  return 1
}

auto_rollback() {
  if [[ "${VIDIOAI_DISABLE_AUTO_ROLLBACK:-false}" == "true" ]]; then
    echo "Rollback automatique désactivé (VIDIOAI_DISABLE_AUTO_ROLLBACK=true)." >&2
    return 0
  fi
  if [[ -x "${PROJECT_DIR}/deploy/scripts/rollback.sh" && -f "${PROJECT_DIR}/.previous-version" ]]; then
    echo "Échec détecté: tentative de rollback automatique..." >&2
    if ! "${PROJECT_DIR}/deploy/scripts/rollback.sh"; then
      echo "Rollback automatique échoué." >&2
      return 1
    fi
    return 0
  fi
  echo "Rollback automatique impossible: script ou version précédente absente." >&2
  echo "Nettoyage automatique de la stack en échec..." >&2
  compose down --remove-orphans || true
  return 1
}

on_error() {
  dump_diagnostics
  auto_rollback || true
}

trap on_error ERR

cd "${PROJECT_DIR}"
test -f "${ENV_FILE}" || { echo "Configuration absente: ${ENV_FILE}" >&2; exit 1; }
test -f "${COMPOSE_FILE}" || { echo "Compose absent: ${COMPOSE_FILE}" >&2; exit 1; }
[[ -n "${VERSION}" && "${VERSION}" != "latest" ]] || { echo "Une version immuable est obligatoire." >&2; exit 1; }
CALLER_SCRATCH_SET=${VIDIOAI_SCRATCH_DIR+x}
CALLER_SCRATCH_VALUE=${VIDIOAI_SCRATCH_DIR-}
set -a
source "${ENV_FILE}"
set +a
if [[ -n "${CALLER_SCRATCH_SET}" && "${CALLER_SCRATCH_VALUE}" != "${VIDIOAI_SCRATCH_DIR:-}" ]]; then
  echo "VIDIOAI_SCRATCH_DIR du shell contredit ${ENV_FILE}; déploiement refusé." >&2
  exit 1
fi
vidioai_require_production_scratch "${ENV_FILE}"

prepare_scratch_permissions() {
  local directories=(models cache work worker-work)
  if [[ -n "${VIDIOAI_TEST_STATE_DIR:-}" ]]; then
    return 0
  fi
  [[ "${EUID}" -eq 0 ]] || {
    echo "Le déploiement doit préparer les ACL Scratch en root." >&2
    return 1
  }
  command -v setfacl >/dev/null 2>&1 || {
    echo "setfacl est requis pour partager le Scratch entre les UID 10001 et 10002." >&2
    return 1
  }
  install -d -m 0770 -o root -g docker \
    "${VIDIOAI_SCRATCH_DIR}/models" \
    "${VIDIOAI_SCRATCH_DIR}/cache" \
    "${VIDIOAI_SCRATCH_DIR}/work" \
    "${VIDIOAI_SCRATCH_DIR}/worker-work"
  setfacl -Rm u:10001:rwx,u:10002:rwx \
    "${VIDIOAI_SCRATCH_DIR}/models" \
    "${VIDIOAI_SCRATCH_DIR}/cache" \
    "${VIDIOAI_SCRATCH_DIR}/work" \
    "${VIDIOAI_SCRATCH_DIR}/worker-work"
  setfacl -Rdm u:10001:rwx,u:10002:rwx \
    "${VIDIOAI_SCRATCH_DIR}/models" \
    "${VIDIOAI_SCRATCH_DIR}/cache" \
    "${VIDIOAI_SCRATCH_DIR}/work" \
    "${VIDIOAI_SCRATCH_DIR}/worker-work"
}

prepare_scratch_permissions

prepare_model_pack_registry_permissions() {
  if [[ -n "${VIDIOAI_TEST_STATE_DIR:-}" ]]; then
    return 0
  fi
  [[ "${EUID}" -eq 0 ]] || {
    echo "Le déploiement doit préparer le registre ModelPack persistant en root." >&2
    return 1
  }
  command -v setfacl >/dev/null 2>&1 || {
    echo "setfacl est requis pour partager le registre ModelPack avec le worker." >&2
    return 1
  }
  local registry_dir="${VIDIOAI_STATE_DIR:-/var/lib/vidioai/state}/model-pack-registry"
  install -d -m 0770 -o root -g docker "${registry_dir}"
  # Le backend publie atomiquement; le worker ne reçoit qu'un bind mount RO,
  # mais a besoin de traverser et lire les versions actives.
  setfacl -Rm u:10001:rwx,u:10002:rx "${registry_dir}"
  setfacl -Rdm u:10001:rwx,u:10002:rx "${registry_dir}"
}

prepare_model_pack_registry_permissions
mkdir -p "${BACKUP_DIR}"
cp "${ENV_FILE}" "${BACKUP_DIR}/env-$(date +%Y%m%d-%H%M%S)"

if [[ "${VIDIOAI_RUN_PREFLIGHT:-true}" != "true" && -z "${VIDIOAI_TEST_STATE_DIR:-}" ]]; then
  echo "Le preflight Scratch est obligatoire en production." >&2
  exit 1
fi
if [[ "${VIDIOAI_RUN_PREFLIGHT:-true}" == "true" ]]; then
  VIDIOAI_PREFLIGHT_SKIP_TESTS=true \
    "${PROJECT_DIR}/deploy/scripts/preflight.sh" "${VERSION}"
fi

# Le backend ne doit jamais démarrer en GPU_PRODUCTION sur les métriques du
# conteneur. Le service natif et son contrat sont donc contrôlés avant le pull.
if [[ "${VIDIOAI_SKIP_HOST_AGENT_CHECK:-false}" != "true" ]]; then
  systemctl is-active --quiet vidioai-host-agent.service \
    || { echo "vidioai-host-agent.service n'est pas actif." >&2; exit 1; }
  curl -fsS -H "X-VidioAI-Host-Token: ${HOST_AGENT_TOKEN}" \
    http://127.0.0.1:8091/system | jq -e '.source == "host"' >/dev/null
fi

if [[ -f .current-version ]]; then
  cp .current-version .previous-version
  curl -fsS -X POST -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
    "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}/api/admin/drain" >/dev/null || true
fi

export VIDIOAI_VERSION="${VERSION}"
compose pull

# Démarrage explicite ordonné pour éviter les états intermédiaires silencieux.
for service in comfyui worker backend frontend proxy; do
  if has_service "${service}"; then
    compose up -d --remove-orphans "${service}"
    wait_for_service "${service}"
    if [[ "${service}" == "worker" && -z "${VIDIOAI_TEST_STATE_DIR:-}" ]]; then
      VIDIOAI_PROJECT_DIR="${PROJECT_DIR}" \
        VIDIOAI_ENV_FILE="${ENV_FILE}" \
        VIDIOAI_COMPOSE_FILE="${COMPOSE_FILE}" \
        "${PROJECT_DIR}/deploy/scripts/verify-scratch.sh" worker
    fi
  fi
done
verify_stack_healthy

if [[ "${VIDIOAI_SKIP_SMOKE_TEST:-false}" != "true" ]]; then
  "${PROJECT_DIR}/deploy/scripts/smoke-test.sh" "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}"
fi

check_proxy_route "/api/health" "200"
check_proxy_route "/api/models/installed" "200"
check_proxy_route "/" "200"
check_proxy_route "/models" "200"

resume_payload=$(curl -fsS -X POST \
  -H "Authorization: Bearer ${VIDIOAI_ADMIN_TOKEN}" \
  "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}/api/admin/resume")
jq -e '.mode == "ACCEPTING_JOBS"' <<<"${resume_payload}" >/dev/null || {
  echo "Le backend n'a pas confirmé ACCEPTING_JOBS après déploiement." >&2
  exit 1
}
ready_payload=$(curl -fsS "http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}/api/ready")
jq -e '.ready == true and .mode == "ACCEPTING_JOBS"' <<<"${ready_payload}" >/dev/null || {
  echo "État final invalide: ready=true et ACCEPTING_JOBS sont obligatoires." >&2
  exit 1
}

printf '%s\n' "${VERSION}" > .current-version
echo "Déploiement ${VERSION} validé."
