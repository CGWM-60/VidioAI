#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
RELEASE_VERSION=${1:-${VIDIOAI_VERSION:-}}
SKIP_TESTS=${VIDIOAI_PREFLIGHT_SKIP_TESTS:-false}

FAILURES=()
WARNINGS=()

fail() {
  FAILURES+=("$1")
}

warn() {
  WARNINGS+=("$1")
}

env_value() {
  local key=${1:?key requis}
  if [[ ! -f "${ENV_FILE}" ]]; then
    return 0
  fi
  sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1
}

is_secret_valid() {
  local value=${1:-}
  [[ -n "${value}" && ${#value} -ge 32 && "${value}" != replace-with-* ]]
}

echo "[preflight] Projet: ${PROJECT_DIR}"

[[ -d "${PROJECT_DIR}" ]] || fail "Répertoire projet introuvable: ${PROJECT_DIR}"
[[ -f "${COMPOSE_FILE}" ]] || fail "Compose production absent: ${COMPOSE_FILE}"
[[ -f "${PROJECT_DIR}/deploy/nginx/default.conf" ]] || fail "Config Nginx absente"
[[ -f "${PROJECT_DIR}/deploy/scripts/deploy.sh" ]] || fail "Script deploy.sh absent"
[[ -f "${PROJECT_DIR}/deploy/scripts/rollback.sh" ]] || fail "Script rollback.sh absent"
[[ -f "${PROJECT_DIR}/deploy/scripts/smoke-test.sh" ]] || fail "Script smoke-test.sh absent"
[[ -f "${PROJECT_DIR}/deploy/scripts/bootstrap-server.sh" ]] || fail "Script bootstrap-server.sh absent"
[[ -f "${PROJECT_DIR}/deploy/scripts/shutdown.sh" ]] || fail "Script shutdown.sh absent"
[[ -f "${PROJECT_DIR}/deploy/scripts/gpu-acceptance.sh" ]] || fail "Script gpu-acceptance.sh absent"

if [[ ! -f "${ENV_FILE}" ]]; then
  fail "Fichier d'environnement absent: ${ENV_FILE}"
else
  worker_token=$(env_value "VIDIOAI_WORKER_TOKEN")
  admin_token=$(env_value "VIDIOAI_ADMIN_TOKEN")
  host_agent_token=$(env_value "HOST_AGENT_TOKEN")

  is_secret_valid "${worker_token}" || fail "VIDIOAI_WORKER_TOKEN invalide (>=32 chars, non placeholder)."
  is_secret_valid "${admin_token}" || fail "VIDIOAI_ADMIN_TOKEN invalide (>=32 chars, non placeholder)."
  is_secret_valid "${host_agent_token}" || fail "HOST_AGENT_TOKEN invalide (>=32 chars, non placeholder)."

  if [[ -n "${worker_token}" && -n "${admin_token}" && "${worker_token}" == "${admin_token}" ]]; then
    fail "VIDIOAI_WORKER_TOKEN et VIDIOAI_ADMIN_TOKEN doivent être différents."
  fi
  if [[ -n "${worker_token}" && -n "${host_agent_token}" && "${worker_token}" == "${host_agent_token}" ]]; then
    fail "VIDIOAI_WORKER_TOKEN et HOST_AGENT_TOKEN doivent être différents."
  fi
  if [[ -n "${admin_token}" && -n "${host_agent_token}" && "${admin_token}" == "${host_agent_token}" ]]; then
    fail "VIDIOAI_ADMIN_TOKEN et HOST_AGENT_TOKEN doivent être différents."
  fi

  hf_token_raw=$(env_value "HF_TOKEN")
  hf_token_trimmed=$(printf '%s' "${hf_token_raw}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [[ -n "${hf_token_raw}" && -z "${hf_token_trimmed}" ]]; then
    warn "HF_TOKEN est uniquement constitué d'espaces: le backend ne doit pas envoyer de header Authorization."
  fi
  if [[ -n "${hf_token_trimmed}" && "${hf_token_trimmed}" != hf_* ]]; then
    fail "HF_TOKEN semble invalide (format attendu préfixé hf_*)."
  fi

  s3_enabled=$(env_value "VIDIOAI_S3_ENABLED")
  if [[ "${s3_enabled:-true}" == "true" ]]; then
    [[ -n "$(env_value "AWS_ACCESS_KEY_ID")" ]] || fail "AWS_ACCESS_KEY_ID requis quand VIDIOAI_S3_ENABLED=true."
    [[ -n "$(env_value "AWS_SECRET_ACCESS_KEY")" ]] || fail "AWS_SECRET_ACCESS_KEY requis quand VIDIOAI_S3_ENABLED=true."
    [[ -n "$(env_value "AWS_S3_BUCKET")" ]] || fail "AWS_S3_BUCKET requis quand VIDIOAI_S3_ENABLED=true."
  fi
fi

if grep -nE ':[[:space:]]*latest($|[[:space:]]|"|\x27)' "${COMPOSE_FILE}" >/dev/null; then
  fail "Le compose production contient un tag latest interdit."
fi

if [[ "${VIDIOAI_PLATFORM:-linux/amd64}" != "linux/amd64" ]]; then
  fail "VIDIOAI_PLATFORM doit cibler linux/amd64 pour la release GPU."
fi

if command -v docker >/dev/null 2>&1; then
  if ! docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config >/dev/null 2>&1; then
    fail "docker compose config échoue pour ${COMPOSE_FILE}."
  fi

  if docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config --format json >/tmp/vidioai-compose-config.json 2>/dev/null; then
    if ! jq -e '.services | to_entries | all(.value.image != null and .value.image != "")' /tmp/vidioai-compose-config.json >/dev/null; then
      fail "Au moins un service n'a pas d'image explicite dans le compose configuré."
    fi
  elif [[ "${COMPOSE_FILE}" == *"compose.production.yml"* ]]; then
    # Fallback minimal si --format json n'est pas supporté.
    while IFS= read -r service; do
      if ! awk "BEGIN{in_svc=0;ok=0} /^  ${service}:/{in_svc=1;next} in_svc && /^  [a-zA-Z0-9_-]+:/{in_svc=0} in_svc && /^[[:space:]]+image:[[:space:]]*.+/{ok=1} END{exit ok?0:1}" "${COMPOSE_FILE}"; then
        fail "Service ${service} sans image explicite dans ${COMPOSE_FILE}."
      fi
    done < <(docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" config --services)
  fi

  if ! docker run --rm \
      -v "${PROJECT_DIR}/deploy/nginx/default.conf:/etc/nginx/conf.d/default.conf:ro" \
      nginx:1.27.5-alpine nginx -t >/dev/null 2>&1; then
    fail "nginx -t échoue avec deploy/nginx/default.conf"
  fi
else
  fail "Docker est requis pour valider compose/nginx."
fi

while IFS= read -r script; do
  bash -n "${script}" || fail "Syntaxe shell invalide: ${script}"
done < <(find "${PROJECT_DIR}/deploy/scripts" "${PROJECT_DIR}/scripts" "${PROJECT_DIR}/deploy/scaleway" -type f -name '*.sh' | sort)

if command -v shellcheck >/dev/null 2>&1; then
  while IFS= read -r script; do
    shellcheck -x "${script}" || fail "shellcheck en échec: ${script}"
  done < <(find "${PROJECT_DIR}/deploy/scripts" "${PROJECT_DIR}/scripts" "${PROJECT_DIR}/deploy/scaleway" -type f -name '*.sh' | sort)
else
  warn "shellcheck non installé: vérification shellcheck ignorée."
fi

if [[ "${SKIP_TESTS}" != "true" ]]; then
  (cd "${PROJECT_DIR}" && pytest worker/tests -q) || fail "Tests Python worker en échec."

  if [[ -f "${PROJECT_DIR}/pyproject.toml" || -f "${PROJECT_DIR}/ruff.toml" || -f "${PROJECT_DIR}/.ruff.toml" ]]; then
    if command -v ruff >/dev/null 2>&1; then
      (cd "${PROJECT_DIR}" && ruff check worker) || fail "ruff check en échec."
      (cd "${PROJECT_DIR}" && ruff format --check worker) || fail "ruff format --check en échec."
    else
      warn "Configuration ruff détectée mais ruff n'est pas installé."
    fi
  fi

  (cd "${PROJECT_DIR}/backend" && cargo fmt --all -- --check) || fail "cargo fmt --check backend en échec."
  (cd "${PROJECT_DIR}/backend" && cargo clippy --all-targets --all-features -- -D warnings) || fail "cargo clippy backend en échec."
  (cd "${PROJECT_DIR}/backend" && cargo test --workspace) || fail "cargo test --workspace backend en échec."

  (cd "${PROJECT_DIR}/host-agent" && cargo fmt --all -- --check) || fail "cargo fmt --check host-agent en échec."
  (cd "${PROJECT_DIR}/host-agent" && cargo clippy --all-targets --all-features -- -D warnings) || fail "cargo clippy host-agent en échec."
  (cd "${PROJECT_DIR}/host-agent" && cargo test --workspace) || fail "cargo test --workspace host-agent en échec."

  if node -e "const p=require('./frontend/package.json'); process.exit(p.scripts && p.scripts.test ? 0 : 1)" 2>/dev/null; then
    (cd "${PROJECT_DIR}" && npm --prefix frontend test) || fail "npm test frontend en échec."
  else
    warn "Script npm test absent dans frontend/package.json (vérification ignorée)."
  fi
  (cd "${PROJECT_DIR}" && npm --prefix frontend run lint) || fail "Lint frontend en échec."
  (cd "${PROJECT_DIR}" && npm --prefix frontend run build) || fail "Build frontend en échec."
  (cd "${PROJECT_DIR}" && bash deploy/tests/test-s3-paths.sh) || fail "Tests deploy/tests/test-s3-paths.sh en échec."
  (cd "${PROJECT_DIR}" && VIDIOAI_COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml" VIDIOAI_HTTP_PORT=18080 bash deploy/tests/test-compose-orchestration.sh) || fail "Test orchestration compose non-GPU en échec."
else
  warn "Tests applicatifs ignorés (VIDIOAI_PREFLIGHT_SKIP_TESTS=true)."
fi

if [[ -n "${RELEASE_VERSION}" ]]; then
  RELEASE_DIR="${PROJECT_DIR}/output/release-${RELEASE_VERSION}"
  if [[ -d "${RELEASE_DIR}" ]]; then
    if [[ -f "${RELEASE_DIR}/SHA256SUMS" ]]; then
      (cd "${RELEASE_DIR}" && sha256sum -c SHA256SUMS >/dev/null) || fail "SHA256SUMS invalide pour release-${RELEASE_VERSION}."
    else
      fail "SHA256SUMS absent dans ${RELEASE_DIR}."
    fi
  else
    warn "Release locale absente pour version ${RELEASE_VERSION}: intégrité non vérifiée."
  fi
fi

if (( ${#WARNINGS[@]} > 0 )); then
  echo "[preflight] WARNINGS"
  for warning in "${WARNINGS[@]}"; do
    echo "  - ${warning}"
  done
fi

if (( ${#FAILURES[@]} > 0 )); then
  echo "PREFLIGHT FAILED"
  for failure in "${FAILURES[@]}"; do
    echo "  - ${failure}"
  done
  exit 1
fi

echo "PREFLIGHT OK"
