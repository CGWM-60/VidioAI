#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FORCE_CONTAINER=${VIDIOAI_FORCE_CONTAINER_WORKER_TESTS:-false}
TEST_DATA_DIR=""

cleanup() {
  if [[ -n "${TEST_DATA_DIR}" && -d "${TEST_DATA_DIR}" ]]; then
    rm -rf -- "${TEST_DATA_DIR}"
  fi
}

trap cleanup EXIT

has_local_media_runtime() {
  command -v ffmpeg >/dev/null 2>&1 \
    && command -v ffprobe >/dev/null 2>&1 \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep 'libx264' >/dev/null
}

if [[ "${FORCE_CONTAINER}" != "true" ]] && has_local_media_runtime; then
  echo "[worker-tests] Runtime média local détecté."
  TEST_DATA_DIR=$(mktemp -d)
  python -m pip install --disable-pip-version-check -r "${PROJECT_DIR}/worker/requirements-test.txt"
  (
    cd "${PROJECT_DIR}"
    VIDIOAI_DATA_DIR="${TEST_DATA_DIR}" \
      PYTHONDONTWRITEBYTECODE=1 \
      python -m pytest -q -p no:cacheprovider worker
  )
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "FFmpeg/FFprobe avec libx264 ou Docker est requis pour tester le Worker." >&2
  exit 1
fi

echo "[worker-tests] Runtime média hôte absent: tests isolés dans Docker, sans sudo."
docker run --rm \
  --env DEBIAN_FRONTEND=noninteractive \
  --env DEBCONF_NOWARNINGS=yes \
  --env PIP_DISABLE_PIP_VERSION_CHECK=1 \
  --env PIP_ROOT_USER_ACTION=ignore \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env VIDIOAI_DATA_DIR=/tmp/vidioai-worker \
  --volume "${PROJECT_DIR}:/workspace:ro" \
  --workdir /workspace \
  python:3.12-bookworm \
  sh -ceu '
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null
    ffmpeg -hide_banner -encoders 2>/dev/null | grep "libx264" >/dev/null
    python -m pip install -q -r worker/requirements-test.txt
    python -m pytest -q -p no:cacheprovider worker
  '
