#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEPLOY_SCRIPT="${ROOT_DIR}/deploy/scripts/deploy.sh"

assert_file_exists() {
  local path=${1:?path requis}
  [[ -f "${path}" ]] || {
    echo "Assertion échouée: fichier absent ${path}" >&2
    exit 1
  }
}

assert_file_contains() {
  local path=${1:?path requis}
  local expected=${2:?valeur attendue}
  grep -Fqx "${expected}" "${path}" || {
    echo "Assertion échouée: ${path} ne contient pas exactement '${expected}'" >&2
    cat "${path}" >&2 || true
    exit 1
  }
}

make_mock_env() {
  local scenario=${1:?scenario requis}
  local sandbox=${2:?sandbox requis}

  local project_dir="${sandbox}/project"
  local mock_bin="${sandbox}/mock-bin"
  local state_dir="${sandbox}/state"

  mkdir -p "${project_dir}/deploy/scripts" "${mock_bin}" "${state_dir}"

  cat >"${project_dir}/.env.production" <<'EOF'
VIDIOAI_WORKER_TOKEN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
VIDIOAI_ADMIN_TOKEN=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
HOST_AGENT_TOKEN=cccccccccccccccccccccccccccccccc
VIDIOAI_HTTP_PORT=8080
VIDIOAI_SCRATCH_DIR=/scratch/vidioai
EOF

  cat >"${project_dir}/compose.production.yml" <<'EOF'
services:
  worker:
    image: mock/worker:test
  backend:
    image: mock/backend:test
  frontend:
    image: mock/frontend:test
  proxy:
    image: mock/proxy:test
EOF

  cat >"${project_dir}/deploy/scripts/rollback.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
: "${VIDIOAI_TEST_STATE_DIR:?VIDIOAI_TEST_STATE_DIR requis}"
echo "rollback" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
EOF
  chmod +x "${project_dir}/deploy/scripts/rollback.sh"

  cat >"${mock_bin}/systemctl" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "is-active" ]]; then
  exit 0
fi
exit 0
EOF
  chmod +x "${mock_bin}/systemctl"

  cat >"${mock_bin}/curl" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
url="${*: -1}"
if [[ "${url}" == *"/system" ]]; then
  echo '{"source":"host"}'
  exit 0
fi
if [[ "${url}" == *"/api/admin/drain" ]]; then
  echo '{}'
  exit 0
fi
echo '{}'
EOF
  chmod +x "${mock_bin}/curl"

  cat >"${mock_bin}/docker" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

: "${VIDIOAI_TEST_STATE_DIR:?VIDIOAI_TEST_STATE_DIR requis}"
services=(worker backend frontend proxy)

set_service_state() {
  local service=${1:?service requis}
  local status=${2:?status requis}
  local health=${3:?health requis}
  local counter_file="${VIDIOAI_TEST_STATE_DIR}/counter-${service}.txt"
  local counter=0
  if [[ -f "${counter_file}" ]]; then
    counter=$(<"${counter_file}")
  fi
  counter=$((counter + 1))
  printf '%s\n' "${counter}" >"${counter_file}"
  local cid="${service}-${counter}"
  printf '%s\n' "${cid}" >"${VIDIOAI_TEST_STATE_DIR}/${service}.cid"
  printf '%s\n' "${status}" >"${VIDIOAI_TEST_STATE_DIR}/${service}.status"
  printf '%s\n' "${health}" >"${VIDIOAI_TEST_STATE_DIR}/${service}.health"
  printf '%s\n' "${service}" >"${VIDIOAI_TEST_STATE_DIR}/${cid}.service"
}

service_from_cid() {
  local cid=${1:?cid requis}
  if [[ -f "${VIDIOAI_TEST_STATE_DIR}/${cid}.service" ]]; then
    cat "${VIDIOAI_TEST_STATE_DIR}/${cid}.service"
    return 0
  fi
  return 1
}

if [[ "${1:-}" == "compose" ]]; then
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--env-file|-p)
        shift 2
        ;;
      config|pull|up|down|ps|logs)
        break
        ;;
      *)
        shift
        ;;
    esac
  done

  cmd=${1:-}
  shift || true

  case "${cmd}" in
    config)
      if [[ "${1:-}" == "--services" ]]; then
        printf '%s\n' "${services[@]}"
        exit 0
      fi
      exit 0
      ;;
    pull)
      echo "pull"
      exit 0
      ;;
    up)
      service_name=""
      for arg in "$@"; do
        case "${arg}" in
          -*) ;;
          *) service_name="${arg}" ;;
        esac
      done
      if [[ -z "${service_name}" ]]; then
        for svc in "${services[@]}"; do
          set_service_state "${svc}" "running" "healthy"
        done
      else
        if [[ "${service_name}" == "frontend" && "${VIDIOAI_TEST_SCENARIO:-}" == "frontend_created_fail" ]]; then
          set_service_state "frontend" "created" "none"
        else
          set_service_state "${service_name}" "running" "healthy"
        fi
      fi
      echo "up ${service_name:-all}" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
      exit 0
      ;;
    down)
      echo "down" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
      exit 0
      ;;
    ps)
      if [[ "${1:-}" == "-q" ]]; then
        svc=${2:-}
        if [[ -f "${VIDIOAI_TEST_STATE_DIR}/${svc}.cid" ]]; then
          cat "${VIDIOAI_TEST_STATE_DIR}/${svc}.cid"
        fi
        exit 0
      fi
      echo "mock-compose-ps"
      exit 0
      ;;
    logs)
      echo "mock-compose-logs"
      exit 0
      ;;
    *)
      echo "Commande compose non supportée: ${cmd}" >&2
      exit 1
      ;;
  esac
fi

if [[ "${1:-}" == "inspect" ]]; then
  shift
  if [[ "${1:-}" == "-f" ]]; then
    format=${2:-}
    cid=${3:-}
    svc=$(service_from_cid "${cid}" || true)
    if [[ -z "${svc}" ]]; then
      echo "unknown"
      exit 0
    fi
    status=$(<"${VIDIOAI_TEST_STATE_DIR}/${svc}.status")
    health=$(<"${VIDIOAI_TEST_STATE_DIR}/${svc}.health")
    if [[ "${format}" == *".State.Status"* ]]; then
      echo "${status}"
      exit 0
    fi
    if [[ "${format}" == *".State.Health"* ]]; then
      echo "${health}"
      exit 0
    fi
    echo "{}"
    exit 0
  fi
  if [[ "${2:-}" == "--format" ]]; then
    echo '{"Status":"created"}'
    exit 0
  fi
  echo '{}'
  exit 0
fi

echo "Commande docker non supportée: $*" >&2
exit 1
EOF
  chmod +x "${mock_bin}/docker"

  echo "${project_dir}|${mock_bin}|${state_dir}|${scenario}"
}

run_deploy_case() {
  local scenario=${1:?scenario requis}
  local expected_exit=${2:?expected_exit requis}
  local version=${3:?version requise}
  local with_current=${4:-false}
  local with_previous=${5:-false}

  local sandbox
  sandbox=$(mktemp -d /tmp/vidioai-deploy-test.XXXXXX)

  IFS='|' read -r project_dir mock_bin state_dir scenario_name <<<"$(make_mock_env "${scenario}" "${sandbox}")"

  if [[ "${with_current}" == "true" ]]; then
    printf 'A\n' >"${project_dir}/.current-version"
  fi
  if [[ "${with_previous}" == "true" ]]; then
    printf 'A\n' >"${project_dir}/.previous-version"
  fi

  local exit_code=0
  set +e
  PATH="${mock_bin}:${PATH}" \
    VIDIOAI_TEST_STATE_DIR="${state_dir}" \
    VIDIOAI_TEST_SCENARIO="${scenario_name}" \
    VIDIOAI_PROJECT_DIR="${project_dir}" \
    VIDIOAI_ENV_FILE="${project_dir}/.env.production" \
    VIDIOAI_COMPOSE_FILE="${project_dir}/compose.production.yml" \
    VIDIOAI_BACKUP_DIR="${state_dir}/backups" \
    VIDIOAI_RUN_PREFLIGHT=false \
    VIDIOAI_SKIP_HOST_AGENT_CHECK=true \
    VIDIOAI_SKIP_SMOKE_TEST=true \
    VIDIOAI_DEPLOY_WAIT_TIMEOUT=1 \
    "${DEPLOY_SCRIPT}" "${version}" >"${state_dir}/deploy.log" 2>&1
  exit_code=$?
  set -e

  if [[ "${expected_exit}" == "success" ]]; then
    [[ "${exit_code}" -eq 0 ]] || {
      echo "Le déploiement devait réussir mais a échoué (code ${exit_code})." >&2
      cat "${state_dir}/deploy.log" >&2 || true
      cat "${state_dir}/events.log" >&2 || true
      exit 1
    }
  else
    [[ "${exit_code}" -ne 0 ]] || {
      echo "Le déploiement devait échouer mais a réussi." >&2
      exit 1
    }
  fi

  echo "${project_dir}|${state_dir}"
}

# 1) Échec frontend en état Created => rollback automatique si version précédente disponible.
IFS='|' read -r project_fail state_fail <<<"$(run_deploy_case frontend_created_fail failure B true false)"
assert_file_exists "${state_fail}/events.log"
grep -Fq "rollback" "${state_fail}/events.log" || {
  echo "Rollback automatique non déclenché sur échec frontend Created." >&2
  cat "${state_fail}/events.log" >&2
  exit 1
}

# 2) Échec frontend sans rollback possible => nettoyage down --remove-orphans.
IFS='|' read -r _project_norb state_norb <<<"$(run_deploy_case frontend_created_fail failure B false false)"
assert_file_exists "${state_norb}/events.log"
grep -Fq "down" "${state_norb}/events.log" || {
  echo "Nettoyage compose down non exécuté quand rollback impossible." >&2
  cat "${state_norb}/events.log" >&2
  exit 1
}

# 3) Idempotence version immuable: même version déployée deux fois doit rester saine.
sandbox_ok=$(mktemp -d /tmp/vidioai-deploy-test.ok.XXXXXX)
trap 'rm -rf "${sandbox_ok}"' EXIT
IFS='|' read -r project_ok mock_ok state_ok scenario_ok <<<"$(make_mock_env stable_success "${sandbox_ok}")"

PATH="${mock_ok}:${PATH}" \
VIDIOAI_TEST_STATE_DIR="${state_ok}" \
VIDIOAI_TEST_SCENARIO="${scenario_ok}" \
VIDIOAI_PROJECT_DIR="${project_ok}" \
VIDIOAI_ENV_FILE="${project_ok}/.env.production" \
VIDIOAI_COMPOSE_FILE="${project_ok}/compose.production.yml" \
VIDIOAI_BACKUP_DIR="${state_ok}/backups" \
VIDIOAI_RUN_PREFLIGHT=false \
VIDIOAI_SKIP_HOST_AGENT_CHECK=true \
VIDIOAI_SKIP_SMOKE_TEST=true \
"${DEPLOY_SCRIPT}" "V1"

PATH="${mock_ok}:${PATH}" \
VIDIOAI_TEST_STATE_DIR="${state_ok}" \
VIDIOAI_TEST_SCENARIO="${scenario_ok}" \
VIDIOAI_PROJECT_DIR="${project_ok}" \
VIDIOAI_ENV_FILE="${project_ok}/.env.production" \
VIDIOAI_COMPOSE_FILE="${project_ok}/compose.production.yml" \
VIDIOAI_BACKUP_DIR="${state_ok}/backups" \
VIDIOAI_RUN_PREFLIGHT=false \
VIDIOAI_SKIP_HOST_AGENT_CHECK=true \
VIDIOAI_SKIP_SMOKE_TEST=true \
"${DEPLOY_SCRIPT}" "V1"

assert_file_contains "${project_ok}/.current-version" "V1"
assert_file_contains "${project_ok}/.previous-version" "V1"

echo "Deploy recovery/idempotence tests: OK"
