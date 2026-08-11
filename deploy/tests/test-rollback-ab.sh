#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROLLBACK_SCRIPT="${ROOT_DIR}/deploy/scripts/rollback.sh"

assert_file_contains() {
  local path=${1:?path requis}
  local expected=${2:?valeur attendue}
  grep -Fqx "${expected}" "${path}" || {
    echo "Assertion échouée: ${path} ne contient pas exactement '${expected}'" >&2
    cat "${path}" >&2 || true
    exit 1
  }
}

sandbox=$(mktemp -d /tmp/vidioai-rollback-test.XXXXXX)
trap 'rm -rf "${sandbox}"' EXIT

project_dir="${sandbox}/project"
mock_bin="${sandbox}/mock-bin"
state_dir="${sandbox}/state"
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

cat >"${project_dir}/deploy/scripts/smoke-test.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
: "${VIDIOAI_TEST_STATE_DIR:?VIDIOAI_TEST_STATE_DIR requis}"
echo "smoke" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
EOF
chmod +x "${project_dir}/deploy/scripts/smoke-test.sh"

printf 'A\n' >"${project_dir}/.previous-version"
printf 'B\n' >"${project_dir}/.current-version"

cat >"${mock_bin}/curl" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
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
  local cid="${service}-ok"
  printf '%s\n' "${cid}" >"${VIDIOAI_TEST_STATE_DIR}/${service}.cid"
  printf '%s\n' "${status}" >"${VIDIOAI_TEST_STATE_DIR}/${service}.status"
  printf '%s\n' "${health}" >"${VIDIOAI_TEST_STATE_DIR}/${service}.health"
  printf '%s\n' "${service}" >"${VIDIOAI_TEST_STATE_DIR}/${cid}.service"
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
      fi
      exit 0
      ;;
    pull)
      echo "pull" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
      exit 0
      ;;
    up)
      svc=""
      for arg in "$@"; do
        [[ "${arg}" == -* ]] || svc="${arg}"
      done
      [[ -n "${svc}" ]] || exit 0
      set_service_state "${svc}" "running" "healthy"
      echo "up ${svc}" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
      exit 0
      ;;
    ps)
      if [[ "${1:-}" == "-q" ]]; then
        svc=${2:-}
        if [[ -f "${VIDIOAI_TEST_STATE_DIR}/${svc}.cid" ]]; then
          cat "${VIDIOAI_TEST_STATE_DIR}/${svc}.cid"
        fi
      fi
      exit 0
      ;;
    logs)
      exit 0
      ;;
    down)
      echo "down" >>"${VIDIOAI_TEST_STATE_DIR}/events.log"
      exit 0
      ;;
    *)
      exit 0
      ;;
  esac
fi

if [[ "${1:-}" == "inspect" && "${2:-}" == "-f" ]]; then
  format=${3:-}
  cid=${4:-}
  svc_file="${VIDIOAI_TEST_STATE_DIR}/${cid}.service"
  if [[ -f "${svc_file}" ]]; then
    svc=$(<"${svc_file}")
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
  fi
  echo "none"
  exit 0
fi

echo '{}'
EOF
chmod +x "${mock_bin}/docker"

PATH="${mock_bin}:${PATH}" \
VIDIOAI_TEST_STATE_DIR="${state_dir}" \
VIDIOAI_PROJECT_DIR="${project_dir}" \
VIDIOAI_ENV_FILE="${project_dir}/.env.production" \
VIDIOAI_COMPOSE_FILE="${project_dir}/compose.production.yml" \
VIDIOAI_DEPLOY_WAIT_TIMEOUT=1 \
"${ROLLBACK_SCRIPT}" >/tmp/vidioai-rollback-test.log 2>&1

assert_file_contains "${project_dir}/.current-version" "A"
assert_file_contains "${project_dir}/.previous-version" "B"
grep -Fq 'smoke' "${state_dir}/events.log" || {
  echo "Le rollback n'a pas exécuté le smoke test." >&2
  cat /tmp/vidioai-rollback-test.log >&2 || true
  cat "${state_dir}/events.log" >&2 || true
  exit 1
}

echo "Rollback A/B test: OK"
