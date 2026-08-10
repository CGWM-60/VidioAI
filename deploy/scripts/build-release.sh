#!/usr/bin/env bash
set -Eeuo pipefail

# Construit et publie les trois images applicatives. À exécuter en CI ou sur une
# machine de build, jamais sur l'Instance GPU de production.
PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "${PROJECT_DIR}/deploy/scripts/lib/s3-paths.sh"
VERSION=${1:?Usage: build-release.sh <version-immuable>}
REGISTRY=${VIDIOAI_REGISTRY:?VIDIOAI_REGISTRY requis}
PLATFORM=${VIDIOAI_PLATFORM:-linux/amd64}

if [[ "${VERSION}" == "latest" || ! "${VERSION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]+$ ]]; then
  echo "Version immuable invalide: ${VERSION}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
cargo fmt --manifest-path backend/Cargo.toml --all -- --check
cargo clippy --manifest-path backend/Cargo.toml --locked --all-targets -- -D warnings
cargo test --manifest-path backend/Cargo.toml --locked
cargo fmt --manifest-path host-agent/Cargo.toml --all -- --check
cargo clippy --manifest-path host-agent/Cargo.toml --locked --all-targets -- -D warnings
cargo test --manifest-path host-agent/Cargo.toml --locked
npm --prefix frontend ci
npm --prefix frontend run lint
npm --prefix frontend run build
python -m pip install --disable-pip-version-check -r worker/requirements-test.txt
python -m pytest -q worker
bash deploy/tests/test-s3-paths.sh
if [[ "${VIDIOAI_RUN_COMPOSE_TESTS:-true}" == "true" ]]; then
  bash deploy/tests/test-compose-orchestration.sh
fi

for service in backend frontend worker; do
  docker buildx build \
    --platform "${PLATFORM}" \
    --file "${service}/Dockerfile" \
    --tag "${REGISTRY}/${service}:${VERSION}" \
    --push "${service}"
done

RELEASE_DIR="${PROJECT_DIR}/output/release-${VERSION}"
mkdir -p "${RELEASE_DIR}/deploy/bin" "${RELEASE_DIR}/deploy/nginx" \
  "${RELEASE_DIR}/deploy/scripts" "${RELEASE_DIR}/deploy/systemd"
cp compose.production.yml .env.production.example "${RELEASE_DIR}/"
cp deploy/nginx/default.conf "${RELEASE_DIR}/deploy/nginx/"
# Le Host Agent doit correspondre à Linux, même lorsque le bundle est préparé
# depuis un Mac. Le build isolé emploie donc exactement la plateforme Docker de
# la release et ne réutilise jamais un binaire Mach-O local.
docker run --rm --platform "${PLATFORM}" \
  -e CARGO_TARGET_DIR=/tmp/vidioai-host-agent-target \
  -v "${PROJECT_DIR}/host-agent:/src:ro" \
  -v "${RELEASE_DIR}/deploy/bin:/out" \
  -w /src rust:1.96-bookworm \
  sh -c 'cargo build --release --locked && cp /tmp/vidioai-host-agent-target/release/vidioai-host-agent /out/'
cp deploy/systemd/vidioai-host-agent.service "${RELEASE_DIR}/deploy/systemd/"
cp deploy/scripts/{bootstrap-server,deploy,rollback,smoke-test,shutdown,preflight,gpu-acceptance}.sh "${RELEASE_DIR}/deploy/scripts/"

# Les digests sont capturés après publication afin que l'audit puisse relier un
# tag lisible aux couches exactes tirées par Docker.
jq -n --arg version "${VERSION}" --arg registry "${REGISTRY}" \
  --arg backend "$(docker buildx imagetools inspect "${REGISTRY}/backend:${VERSION}" --format '{{json .Manifest.Digest}}' | tr -d '"')" \
  --arg frontend "$(docker buildx imagetools inspect "${REGISTRY}/frontend:${VERSION}" --format '{{json .Manifest.Digest}}' | tr -d '"')" \
  --arg worker "$(docker buildx imagetools inspect "${REGISTRY}/worker:${VERSION}" --format '{{json .Manifest.Digest}}' | tr -d '"')" \
  '{version:$version, registry:$registry, images:{backend:$backend,frontend:$frontend,worker:$worker}}' \
  > "${RELEASE_DIR}/release.json"
(cd "${RELEASE_DIR}" && sha256sum release.json compose.production.yml .env.production.example \
  deploy/bin/vidioai-host-agent deploy/systemd/vidioai-host-agent.service > SHA256SUMS)
tar -C "${PROJECT_DIR}/output" -czf "${PROJECT_DIR}/output/vidioai-release-${VERSION}.tar.gz" "release-${VERSION}"
if [[ -n "${AWS_S3_BUCKET:-}" ]]; then
  vidioai_validate_s3_bucket "${AWS_S3_BUCKET}"
  DEPLOYMENT_URI=$(vidioai_release_uri "${AWS_S3_BUCKET}" "${VERSION}" deployment.tar.gz)
  MANIFEST_URI=$(vidioai_release_uri "${AWS_S3_BUCKET}" "${VERSION}" release.json)
  AWS_ENDPOINT_ARGS=()
  if [[ -n "${AWS_ENDPOINT_URL_S3:-}" ]]; then
    AWS_ENDPOINT_ARGS=(--endpoint-url "${AWS_ENDPOINT_URL_S3}")
  fi
  aws s3 cp "${PROJECT_DIR}/output/vidioai-release-${VERSION}.tar.gz" \
    "${DEPLOYMENT_URI}" \
    --storage-class "${AWS_S3_STORAGE_CLASS:-STANDARD}" \
    "${AWS_ENDPOINT_ARGS[@]}"
  aws s3 cp "${RELEASE_DIR}/release.json" \
    "${MANIFEST_URI}" \
    --storage-class "${AWS_S3_STORAGE_CLASS:-STANDARD}" \
    "${AWS_ENDPOINT_ARGS[@]}"
fi
echo "Release publiée: ${VERSION}"
