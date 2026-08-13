#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/vidioai-release-install-test.XXXXXX")
trap 'rm -rf "${TEST_ROOT}"' EXIT

VERSION=2099.08.13-1
DIGEST=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
FAKE_BIN="${TEST_ROOT}/bin"
INSTALL_DIR="${TEST_ROOT}/install"
RELEASE_DIR="${TEST_ROOT}/release-${VERSION}"
ARCHIVE="${TEST_ROOT}/vidioai-release-${VERSION}.tar.gz"
LOG_FILE="${TEST_ROOT}/calls.log"
mkdir -p "${FAKE_BIN}" "${INSTALL_DIR}/deploy/scripts" "${RELEASE_DIR}/deploy/scripts"

cat > "${FAKE_BIN}/docker" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${1:-}" == "buildx" && "${2:-}" == "imagetools" && "${3:-}" == "inspect" ]]; then
  if [[ "${VIDIOAI_TEST_INSPECT_FAIL:-false}" == "true" ]]; then
    exit 1
  fi
  printf '"%s"\n' "${VIDIOAI_TEST_DIGEST}"
  exit 0
fi
if [[ "${1:-}" == "pull" ]]; then
  printf 'pull:%s\n' "${2}" >> "${VIDIOAI_TEST_LOG}"
  exit 0
fi
if [[ "${1:-}" == "compose" ]]; then
  printf 'worker-health\n' >> "${VIDIOAI_TEST_LOG}"
  printf '{"status":"ok","service":"vidioai-gpu-worker","version":"%s"}\n' "${VIDIOAI_TEST_WORKER_HEALTH_VERSION}"
  exit 0
fi
exit 97
EOF
chmod +x "${FAKE_BIN}/docker"

cat > "${FAKE_BIN}/curl" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
url="${*: -1}"
printf 'health:%s\n' "${url}" >> "${VIDIOAI_TEST_LOG}"
if [[ "${url}" == */api/health ]]; then
  printf '{"status":"ok","version":"%s"}\n' "${VIDIOAI_TEST_BACKEND_HEALTH_VERSION}"
else
  printf '{}\n'
fi
EOF
chmod +x "${FAKE_BIN}/curl"

cat > "${INSTALL_DIR}/deploy/scripts/shutdown.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
printf 'shutdown\n' >> "${VIDIOAI_TEST_LOG}"
EOF
chmod +x "${INSTALL_DIR}/deploy/scripts/shutdown.sh"
printf '%s\n' old-version > "${INSTALL_DIR}/.current-version"

cat > "${INSTALL_DIR}/.env.production" <<EOF
VIDIOAI_VERSION=old-version
VIDIOAI_REGISTRY=registry.old/vidioai
VIDIOAI_HTTP_PORT=8080
EOF

cat > "${RELEASE_DIR}/deploy/scripts/deploy.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
printf 'deploy:%s\n' "${1}" >> "${VIDIOAI_TEST_LOG}"
printf '%s\n' "${1}" > "${VIDIOAI_PROJECT_DIR}/.current-version"
EOF
chmod +x "${RELEASE_DIR}/deploy/scripts/deploy.sh"

cat > "${RELEASE_DIR}/compose.production.yml" <<'EOF'
name: vidioai-release-install-test
services: {}
EOF
cat > "${RELEASE_DIR}/.env.production.example" <<EOF
VIDIOAI_VERSION=${VERSION}
VIDIOAI_REGISTRY=registry.example/vidioai
EOF
cat > "${RELEASE_DIR}/release.json" <<EOF
{
  "version": "${VERSION}",
  "registry": "registry.example/vidioai",
  "images": {
    "backend": "${DIGEST}",
    "frontend": "${DIGEST}",
    "worker": "${DIGEST}"
  }
}
EOF
(
  cd "${RELEASE_DIR}"
  find . -type f ! -path './SHA256SUMS' -print \
    | sed 's#^\./##' \
    | LC_ALL=C sort \
    | while IFS= read -r bundle_file; do
        sha256sum "${bundle_file}"
      done > SHA256SUMS
)
tar -C "${TEST_ROOT}" -czf "${ARCHIVE}" "release-${VERSION}"

OUTPUT=$(PATH="${FAKE_BIN}:${PATH}" \
  VIDIOAI_RELEASE_ARCHIVE="${ARCHIVE}" \
  VIDIOAI_INSTALL_DIR="${INSTALL_DIR}" \
  VIDIOAI_ENV_FILE="${INSTALL_DIR}/.env.production" \
  VIDIOAI_COMPOSE_FILE="${INSTALL_DIR}/compose.production.yml" \
  COMPOSE_PROFILES=comfyui \
  VIDIOAI_COMFYUI_IMAGE="registry.example/comfyui@${DIGEST}" \
  VIDIOAI_TEST_DIGEST="${DIGEST}" \
  VIDIOAI_TEST_BACKEND_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_WORKER_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_LOG="${LOG_FILE}" \
  bash "${PROJECT_DIR}/deploy/release-install.sh" "${VERSION}")

for expected in \
  "[1/7] Version" \
  "[2/7] Download" \
  "[3/7] Verify" \
  "[4/7] Build" \
  "[5/7] Install" \
  "[6/7] Start" \
  "[7/7] Healthcheck"; do
  [[ "${OUTPUT}" == *"${expected}"* ]]
done
[[ "${OUTPUT}" == *"VidioAI ${VERSION} deployed successfully"* ]]
grep -Fxq "VIDIOAI_VERSION=${VERSION}" "${INSTALL_DIR}/.env.production"
grep -Fxq "VIDIOAI_REGISTRY=registry.example/vidioai" "${INSTALL_DIR}/.env.production"
[[ "$(sed -n '1p' "${LOG_FILE}")" == "pull:registry.example/vidioai/backend:${VERSION}" ]]
grep -Fxq shutdown "${LOG_FILE}"
grep -Fxq "deploy:${VERSION}" "${LOG_FILE}"
grep -Fxq "pull:registry.example/comfyui@${DIGEST}" "${LOG_FILE}"
grep -Fxq worker-health "${LOG_FILE}"
[[ "$(grep -nE 'shutdown|deploy:' "${LOG_FILE}" | cut -d: -f2- | tr '\n' ' ')" == "shutdown deploy:${VERSION} " ]]

# Un fichier injecté mais absent de SHA256SUMS doit être refusé pendant Verify,
# avant pull, arrêt ou installation.
TAMPER_ROOT="${TEST_ROOT}/tampered"
TAMPER_ARCHIVE="${TEST_ROOT}/tampered-release.tar.gz"
mkdir -p "${TAMPER_ROOT}"
cp -R "${RELEASE_DIR}" "${TAMPER_ROOT}/"
printf 'unsigned\n' > "${TAMPER_ROOT}/release-${VERSION}/deploy/unsigned-script.sh"
tar -C "${TAMPER_ROOT}" -czf "${TAMPER_ARCHIVE}" "release-${VERSION}"
: > "${LOG_FILE}"
set +e
TAMPER_OUTPUT=$(PATH="${FAKE_BIN}:${PATH}" \
  VIDIOAI_RELEASE_ARCHIVE="${TAMPER_ARCHIVE}" \
  VIDIOAI_INSTALL_DIR="${INSTALL_DIR}" \
  VIDIOAI_ENV_FILE="${INSTALL_DIR}/.env.production" \
  VIDIOAI_COMPOSE_FILE="${INSTALL_DIR}/compose.production.yml" \
  VIDIOAI_TEST_DIGEST="${DIGEST}" \
  VIDIOAI_TEST_BACKEND_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_WORKER_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_LOG="${LOG_FILE}" \
  bash "${PROJECT_DIR}/deploy/release-install.sh" "${VERSION}" 2>&1)
TAMPER_STATUS=$?
set -e
[[ ${TAMPER_STATUS} -ne 0 ]]
[[ "${TAMPER_OUTPUT}" == *"FAILED VidioAI ${VERSION} at [3/7] Verify"* ]]
[[ "${TAMPER_OUTPUT}" == *"la liste des fichiers ne correspond pas à SHA256SUMS"* ]]
[[ ! -s "${LOG_FILE}" ]]

# Un service sain mais construit avec une autre version ne valide jamais le
# déploiement. Le worker et le backend utilisent le même contrat versionné.
: > "${LOG_FILE}"
set +e
VERSION_OUTPUT=$(PATH="${FAKE_BIN}:${PATH}" \
  VIDIOAI_RELEASE_ARCHIVE="${ARCHIVE}" \
  VIDIOAI_INSTALL_DIR="${INSTALL_DIR}" \
  VIDIOAI_ENV_FILE="${INSTALL_DIR}/.env.production" \
  VIDIOAI_COMPOSE_FILE="${INSTALL_DIR}/compose.production.yml" \
  VIDIOAI_TEST_DIGEST="${DIGEST}" \
  VIDIOAI_TEST_BACKEND_HEALTH_VERSION="wrong-version" \
  VIDIOAI_TEST_WORKER_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_LOG="${LOG_FILE}" \
  bash "${PROJECT_DIR}/deploy/release-install.sh" "${VERSION}" 2>&1)
VERSION_STATUS=$?
set -e
[[ ${VERSION_STATUS} -ne 0 ]]
[[ "${VERSION_OUTPUT}" == *"FAILED VidioAI ${VERSION} at [7/7] Healthcheck"* ]]
[[ "${VERSION_OUTPUT}" == *"Le backend sain n'expose pas la version demandée ${VERSION}"* ]]

: > "${LOG_FILE}"
set +e
WORKER_VERSION_OUTPUT=$(PATH="${FAKE_BIN}:${PATH}" \
  VIDIOAI_RELEASE_ARCHIVE="${ARCHIVE}" \
  VIDIOAI_INSTALL_DIR="${INSTALL_DIR}" \
  VIDIOAI_ENV_FILE="${INSTALL_DIR}/.env.production" \
  VIDIOAI_COMPOSE_FILE="${INSTALL_DIR}/compose.production.yml" \
  VIDIOAI_TEST_DIGEST="${DIGEST}" \
  VIDIOAI_TEST_BACKEND_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_WORKER_HEALTH_VERSION="wrong-version" \
  VIDIOAI_TEST_LOG="${LOG_FILE}" \
  bash "${PROJECT_DIR}/deploy/release-install.sh" "${VERSION}" 2>&1)
WORKER_VERSION_STATUS=$?
set -e
[[ ${WORKER_VERSION_STATUS} -ne 0 ]]
[[ "${WORKER_VERSION_OUTPUT}" == *"FAILED VidioAI ${VERSION} at [7/7] Healthcheck"* ]]
[[ "${WORKER_VERSION_OUTPUT}" == *"Le worker sain n'expose pas la version demandée ${VERSION}"* ]]

# Un tag distant absent doit arrêter le wrapper pendant Version, avant pull,
# shutdown, copie du bundle ou invocation du déploiement existant.
: > "${LOG_FILE}"
set +e
FAIL_OUTPUT=$(PATH="${FAKE_BIN}:${PATH}" \
  VIDIOAI_RELEASE_ARCHIVE="${ARCHIVE}" \
  VIDIOAI_INSTALL_DIR="${INSTALL_DIR}" \
  VIDIOAI_ENV_FILE="${INSTALL_DIR}/.env.production" \
  VIDIOAI_COMPOSE_FILE="${INSTALL_DIR}/compose.production.yml" \
  VIDIOAI_TEST_DIGEST="${DIGEST}" \
  VIDIOAI_TEST_INSPECT_FAIL=true \
  VIDIOAI_TEST_BACKEND_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_WORKER_HEALTH_VERSION="${VERSION}" \
  VIDIOAI_TEST_LOG="${LOG_FILE}" \
  bash "${PROJECT_DIR}/deploy/release-install.sh" "${VERSION}" 2>&1)
FAIL_STATUS=$?
set -e
[[ ${FAIL_STATUS} -ne 0 ]]
[[ "${FAIL_OUTPUT}" == *"FAILED VidioAI ${VERSION} at [1/7] Version"* ]]
[[ ! -s "${LOG_FILE}" ]]

echo "Release install orchestration test: OK"
