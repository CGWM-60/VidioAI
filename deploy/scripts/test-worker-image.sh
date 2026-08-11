#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE=${1:?Usage: test-worker-image.sh <worker-image>}
TOKEN=${VIDIOAI_CONTAINER_TEST_TOKEN:-vidioai-container-contract-token}
CONTAINER_NAME="vidioai-worker-contract-$$"
SCRATCH_ROOT=""

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  if [[ -n "${SCRATCH_ROOT}" && -d "${SCRATCH_ROOT}" ]]; then
    # Le Worker (UID 10002) crée notamment runtime-deps avec des répertoires
    # non inscriptibles par l'utilisateur du runner CI. Nettoyer le bind mount
    # depuis l'image en root avant de retirer le dossier hôte évite qu'un test
    # entièrement vert échoue uniquement dans le trap EXIT.
    docker run --rm \
      --user 0 \
      --entrypoint sh \
      -e CLEANUP_UID="$(id -u)" \
      -e CLEANUP_GID="$(id -g)" \
      -v "${SCRATCH_ROOT}:/scratch" \
      "${IMAGE}" \
      -c 'chown -R "${CLEANUP_UID}:${CLEANUP_GID}" /scratch' >/dev/null 2>&1 || true
    rm -rf -- "${SCRATCH_ROOT}" || true
  fi
}
trap cleanup EXIT

command -v docker >/dev/null 2>&1 || {
  echo "Docker est requis pour le contrat de l'image Worker." >&2
  exit 1
}

docker image inspect "${IMAGE}" >/dev/null

# Le test s'exécute dans l'image exacte qui sera publiée. Il ne réutilise ni le
# Python de l'hôte ni requirements-test.txt.
docker run --rm --entrypoint python "${IMAGE}" -c '
import importlib.metadata as metadata
import platform
import diffusers
import bitsandbytes
from bitsandbytes.nn import Linear4bit
import torch
import transformers
from huggingface_hub import HfApi, snapshot_download

expected = {
    "diffusers": "0.39.0",
    "transformers": "5.14.1",
    "accelerate": "1.14.0",
    "huggingface-hub": "1.24.0",
    "bitsandbytes": "0.49.2",
    "sentencepiece": "0.2.2",
    "einops": "0.8.2",
    "ftfy": "6.3.1",
    "imageio": "2.37.4",
    "imageio-ffmpeg": "0.6.0",
}
for package, version in expected.items():
    actual = metadata.version(package)
    assert actual == version, f"{package}: {actual} != {version}"
for class_name in (
    "StableDiffusionPipeline",
    "StableDiffusionXLPipeline",
    "FluxPipeline",
    "CogVideoXPipeline",
    "WanPipeline",
    "HunyuanVideoPipeline",
    "LTXPipeline",
):
    assert getattr(diffusers, class_name, None) is not None, class_name
assert torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert bitsandbytes.__version__ == "0.49.2"
assert Linear4bit is not None
assert platform.machine() in {"x86_64", "amd64"}, platform.machine()
assert transformers.__version__
assert HfApi is not None and snapshot_download is not None
print("WORKER_IMPORT_CONTRACT_OK")
'

docker run --rm --entrypoint sh "${IMAGE}" -ceu '
  command -v ffmpeg >/dev/null
  command -v ffprobe >/dev/null
  ffmpeg -hide_banner -encoders 2>/dev/null | grep "libx264" >/dev/null
'

SCRATCH_ROOT=$(mktemp -d /tmp/vidioai-worker-image.XXXXXX)
mkdir -p "${SCRATCH_ROOT}/models" "${SCRATCH_ROOT}/cache" "${SCRATCH_ROOT}/work" "${SCRATCH_ROOT}/worker-work"
docker run --rm --user 0 --entrypoint sh \
  -v "${SCRATCH_ROOT}:/scratch" "${IMAGE}" \
  -c 'chown -R 10002:10002 /scratch/models /scratch/cache /scratch/work /scratch/worker-work && chmod -R u+rwX,go-rwx /scratch/models /scratch/cache /scratch/work /scratch/worker-work'

docker run -d --name "${CONTAINER_NAME}" \
  -e APP_ENV=GPU_PRODUCTION \
  -e GPU_REQUIRED=true \
  -e VIDIOAI_MIN_SCRATCH_TOTAL_BYTES=1 \
  -e VIDIOAI_WORKER_TOKEN="${TOKEN}" \
  -e VIDIOAI_MODELS_DIR=/models \
  -e HF_HOME=/cache/huggingface \
  -e VIDIOAI_WORK_DIR=/worker-work \
  -e VIDIOAI_OUTPUTS_DIR=/work \
  -v "${SCRATCH_ROOT}/models:/models" \
  -v "${SCRATCH_ROOT}/cache:/cache" \
  -v "${SCRATCH_ROOT}/work:/work" \
  -v "${SCRATCH_ROOT}/worker-work:/worker-work" \
  "${IMAGE}" >/dev/null

for _attempt in $(seq 1 40); do
  if docker exec "${CONTAINER_NAME}" curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

docker exec "${CONTAINER_NAME}" curl -fsS http://127.0.0.1:8000/health \
  | jq -e '.status == "ok" and .service == "vidioai-gpu-worker"' >/dev/null

unauthorized_status=$(docker exec "${CONTAINER_NAME}" \
  curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/ready)
[[ "${unauthorized_status}" == "401" ]] || {
  echo "/ready sans token a retourné HTTP ${unauthorized_status}, attendu 401." >&2
  exit 1
}

ready_file="${SCRATCH_ROOT}/ready.json"
ready_status=$(docker exec "${CONTAINER_NAME}" \
  curl -sS -o /tmp/ready.json -w '%{http_code}' \
  -H "X-VidioAI-Worker-Token: ${TOKEN}" http://127.0.0.1:8000/ready)
[[ "${ready_status}" == "200" || "${ready_status}" == "503" ]] || {
  echo "/ready authentifié a retourné HTTP ${ready_status}, attendu 200 ou 503." >&2
  docker logs "${CONTAINER_NAME}" >&2
  exit 1
}
docker cp "${CONTAINER_NAME}:/tmp/ready.json" "${ready_file}" >/dev/null
jq -e '
  (.ready | type == "boolean") and
  (.runtime_available | type == "boolean") and
  (.cuda_available | type == "boolean") and
  (.scratch_mount_ok == true) and
  (.scratch_filesystem | type == "string") and
  (.scratch_total_bytes > 0) and
  (.scratch_available_bytes > 0) and
  (.errors | type == "array")
' "${ready_file}" >/dev/null

docker exec "${CONTAINER_NAME}" python -c '
from pathlib import Path
for raw in ("/models", "/cache", "/work", "/worker-work"):
    path = Path(raw) / ".vidioai-contract"
    path.write_text("ok", encoding="utf-8")
    path.unlink()
print("WORKER_MOUNTS_WRITABLE_OK")
'

echo "WORKER_IMAGE_CONTRACT_OK image=${IMAGE} ready_http=${ready_status}"
