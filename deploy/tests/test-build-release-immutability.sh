#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FAKE_BIN=$(mktemp -d)
trap 'rm -rf "${FAKE_BIN}"' EXIT

cat > "${FAKE_BIN}/docker" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == buildx\ imagetools\ inspect* ]]; then
  exit 0
fi
exit 99
EOF
chmod +x "${FAKE_BIN}/docker"

set +e
OUTPUT=$(PATH="${FAKE_BIN}:${PATH}" \
  VIDIOAI_REGISTRY=registry.example/vidioai \
  bash "${PROJECT_DIR}/deploy/scripts/build-release.sh" 2099.01.01-immutable-test 2>&1)
STATUS=$?
set -e

[[ ${STATUS} -ne 0 ]]
[[ "${OUTPUT}" == *"le tag existe déjà"* ]]

echo "Build release immutability test: OK"
