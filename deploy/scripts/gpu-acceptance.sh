#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
BASE_URL=${1:-http://127.0.0.1:8080}
TARGET_VERSION=${VIDIOAI_GPU_ACCEPTANCE_VERSION:-2026.08.11-12}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
source "${PROJECT_DIR}/deploy/scripts/lib/scratch-storage.sh"
CALLER_SCRATCH_SET=${VIDIOAI_SCRATCH_DIR+x}
CALLER_SCRATCH_VALUE=${VIDIOAI_SCRATCH_DIR-}
set -a
source "${ENV_FILE}"
set +a
if [[ -n "${CALLER_SCRATCH_SET}" && "${CALLER_SCRATCH_VALUE}" != "${VIDIOAI_SCRATCH_DIR:-}" ]]; then
  echo "VIDIOAI_SCRATCH_DIR du shell contredit ${ENV_FILE}; acceptance refusée." >&2
  exit 1
fi
vidioai_require_production_scratch "${ENV_FILE}"
MODEL_ID=${VIDIOAI_GPU_ACCEPTANCE_MODEL_ID:?VIDIOAI_GPU_ACCEPTANCE_MODEL_ID organisation/modèle requis}
WORKER_TOKEN=${VIDIOAI_WORKER_TOKEN:?VIDIOAI_WORKER_TOKEN requis}
ATTEMPTS=${VIDIOAI_GPU_ACCEPTANCE_ATTEMPTS:-240}
RESULT_ROOT=${VIDIOAI_GPU_ACCEPTANCE_RESULT_DIR:-${PROJECT_DIR}/output/gpu-acceptance-${TARGET_VERSION}}
TEMP_ROOT=""

compose() {
  VIDIOAI_SCRATCH_DIR="${VIDIOAI_SCRATCH_DIR}" \
    docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" "$@"
}

diagnostics() {
  echo "=== GPU ACCEPTANCE DIAGNOSTICS ===" >&2
  nvidia-smi >&2 || true
  compose ps -a >&2 || true
  compose logs --tail=240 worker backend proxy >&2 || true
  local worker_id
  worker_id=$(compose ps -q worker 2>/dev/null || true)
  if [[ -n "${worker_id}" ]]; then
    docker inspect "${worker_id}" --format '{{json .State}}' >&2 || true
    docker exec "${worker_id}" python -c '
import torch
print({
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
' >&2 || true
  fi
}

cleanup() {
  if [[ -n "${TEMP_ROOT}" && -d "${TEMP_ROOT}" ]]; then
    rm -rf -- "${TEMP_ROOT}"
  fi
}
trap diagnostics ERR
trap cleanup EXIT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Commande requise manquante: $1" >&2
    exit 1
  }
}

for command in curl jq ffmpeg ffprobe docker nvidia-smi; do
  require_cmd "${command}"
done

[[ -f "${PROJECT_DIR}/.current-version" ]] || {
  echo "Version déployée introuvable: ${PROJECT_DIR}/.current-version" >&2
  exit 1
}
deployed_version=$(tr -d '[:space:]' < "${PROJECT_DIR}/.current-version")
[[ "${deployed_version}" == "${TARGET_VERSION}" ]] || {
  echo "Release déployée ${deployed_version}, attendue ${TARGET_VERSION}." >&2
  exit 1
}

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)
[[ "${gpu_name}" == *"NVIDIA L40S"* || "${gpu_name}" == *"L40S"* ]] || {
  echo "GPU inattendu: ${gpu_name}; NVIDIA L40S requise." >&2
  exit 1
}

mkdir -p "${RESULT_ROOT}"
worker_id=$(compose ps -q worker)
[[ -n "${worker_id}" ]] || { echo "Conteneur Worker absent." >&2; exit 1; }
VIDIOAI_PROJECT_DIR="${PROJECT_DIR}" VIDIOAI_ENV_FILE="${ENV_FILE}" \
  VIDIOAI_COMPOSE_FILE="${COMPOSE_FILE}" \
  "${PROJECT_DIR}/deploy/scripts/verify-scratch.sh" worker \
  | tee "${RESULT_ROOT}/scratch.json" >/dev/null
docker exec "${worker_id}" python -c '
import torch
assert torch.cuda.is_available(), "CUDA indisponible"
assert "L40S" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
' | tee "${RESULT_ROOT}/worker-runtime.txt"
docker exec "${worker_id}" sh -ceu '
  command -v ffmpeg >/dev/null
  command -v ffprobe >/dev/null
  ffmpeg -hide_banner -encoders 2>/dev/null | grep "libx264" >/dev/null
  for directory in /models /cache /worker-work /work; do
    probe="${directory}/.vidioai-gpu-acceptance"
    printf ok > "${probe}"
    rm -f "${probe}"
  done
'

worker_ready_status=$(docker exec "${worker_id}" curl -sS -o /tmp/vidioai-ready.json -w '%{http_code}' \
  -H "X-VidioAI-Worker-Token: ${WORKER_TOKEN}" http://127.0.0.1:8000/ready)
[[ "${worker_ready_status}" == "200" ]] || {
  docker exec "${worker_id}" cat /tmp/vidioai-ready.json >&2 || true
  echo "Worker /ready HTTP ${worker_ready_status}, attendu 200." >&2
  exit 1
}
docker cp "${worker_id}:/tmp/vidioai-ready.json" "${RESULT_ROOT}/worker-ready.json" >/dev/null
jq -e '.ready and .runtime_available and .cuda_available and .scratch_mount_ok and
  (.scratch_total_bytes > 214748364800) and (.scratch_available_bytes > 0)' \
  "${RESULT_ROOT}/worker-ready.json" >/dev/null

curl -fsS "${BASE_URL}/api/ready" | tee "${RESULT_ROOT}/backend-ready.json" \
  | jq -e '.ready and .worker and .runtime and .gpu and .scratch_mount_ok and
      (.scratch_total_bytes > 214748364800) and (.scratch_available_bytes > 0)' >/dev/null

poll_job() {
  local job_id=${1:?job id requis}
  for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
    local payload status
    payload=$(curl -fsS "${BASE_URL}/api/jobs/${job_id}")
    status=$(jq -r '.status | ascii_downcase' <<<"${payload}")
    if [[ "${status}" == "completed" ]]; then
      printf '%s\n' "${payload}"
      return 0
    fi
    if [[ "${status}" == "failed" || "${status}" == "cancelled" || "${status}" == "interrupted" ]]; then
      jq . <<<"${payload}" >&2
      return 1
    fi
    sleep 3
  done
  echo "Timeout du job ${job_id}." >&2
  return 1
}

poll_generation() {
  local generation_id=${1:?generation id requis}
  for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
    local payload status
    payload=$(curl -fsS "${BASE_URL}/api/generations/${generation_id}")
    status=$(jq -r '.status | ascii_downcase' <<<"${payload}")
    if [[ "${status}" == "completed" ]]; then
      printf '%s\n' "${payload}"
      return 0
    fi
    if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      jq . <<<"${payload}" >&2
      return 1
    fi
    sleep 3
  done
  echo "Timeout de la génération ${generation_id}." >&2
  return 1
}

curl -fsS "${BASE_URL}/api/ready" \
  | jq -e '.ready == true and .mode == "ACCEPTING_JOBS"' >/dev/null
curl -fsS "${BASE_URL}/api/resources" | tee "${RESULT_ROOT}/resources-before-generation.json" \
  | jq -e '.worker.gpu.vram_total_bytes > 0 and .worker.memory.ram_available_bytes > 0' >/dev/null
curl -fsS "${BASE_URL}/api/models/installed" \
  | jq -e '.items | type == "array"' >/dev/null

model=$(curl -fsS --get --data-urlencode "model_id=${MODEL_ID}" "${BASE_URL}/api/models/by-id")
jq -e '
  .source_available and .hardware_compatible and .installable and
  ((.runtime_compatibility == "SUPPORTED") or (.runtime_compatibility == "UNKNOWN")) and
  (.capabilities | index("TEXT_TO_VIDEO")) and
  (.capabilities | index("IMAGE_TO_VIDEO"))
' <<<"${model}" >/dev/null

if ! jq -e '.installed' <<<"${model}" >/dev/null; then
  job_id=$(curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg model_id "${MODEL_ID}" --arg revision "$(jq -r '.revision' <<<"${model}")" '{model_id:$model_id, revision:$revision}')" \
    "${BASE_URL}/api/models/install" | jq -er '.id')
  poll_job "${job_id}" | tee "${RESULT_ROOT}/install-job.json" >/dev/null
fi

model=$(curl -fsS --get --data-urlencode "model_id=${MODEL_ID}" "${BASE_URL}/api/models/by-id")
if ! jq -e '.runtime_ready' <<<"${model}" >/dev/null; then
  curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg model_id "${MODEL_ID}" '{model_id:$model_id}')" \
    "${BASE_URL}/api/models/load" | tee "${RESULT_ROOT}/load.json" >/dev/null
fi

model=$(curl -fsS --get --data-urlencode "model_id=${MODEL_ID}" "${BASE_URL}/api/models/by-id")
jq -e '.installed and .loaded and .runtime_ready' <<<"${model}" >/dev/null

TEMP_ROOT=$(mktemp -d /tmp/vidioai-gpu-acceptance.XXXXXX)
ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=blue:s=640x360:d=1 \
  -frames:v 1 "${TEMP_ROOT}/source.png"
source_asset=$(curl -fsS -X POST -F "file=@${TEMP_ROOT}/source.png;type=image/png" \
  "${BASE_URL}/api/assets" | jq -er '.id')

validate_video() {
  local label=${1:?label requis}
  local asset_id=${2:?asset id requis}
  local quality=${3:?quality requise}
  local output="${RESULT_ROOT}/${label}.mp4"
  local probe="${RESULT_ROOT}/${label}.ffprobe.json"
  curl -fsS "${BASE_URL}/api/assets/${asset_id}" -o "${output}"
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=codec_name,width,height,nb_frames,nb_read_frames,avg_frame_rate:format=duration \
    -of json "${output}" | tee "${probe}" >/dev/null
  jq -e '
    .streams[0].codec_name == "h264" and
    (.streams[0].width > 0) and (.streams[0].height > 0) and
    ((.format.duration | tonumber) > 0) and
    (((.streams[0].nb_read_frames // .streams[0].nb_frames) | tonumber) > 1)
  ' "${probe}" >/dev/null
  echo "GPU_VIDEO_OK label=${label} quality=${quality} asset=${asset_id}"
}

run_video() {
  local label=${1:?label requis}
  local capability=${2:?capability requise}
  local mode=${3:?mode requis}
  local quality=${4:?quality requise}
  local input_asset=${5:-}
  local body generation_id generation asset_id
  body=$(jq -n \
    --arg model_id "${MODEL_ID}" \
    --arg capability "${capability}" \
    --arg mode "${mode}" \
    --arg quality "${quality}" \
    --arg input_asset_id "${input_asset}" \
    '{
      model_id:$model_id,
      capability:$capability,
      mode:$mode,
      prompt:"A short cinematic camera movement over a futuristic city",
      quality:$quality,
      aspect_ratio:"16:9",
      duration_seconds:2,
      fps:12,
      input_asset_id:(if ($input_asset_id|length) > 0 then $input_asset_id else null end),
      input_images:(if ($input_asset_id|length) > 0 then [{asset_id:$input_asset_id,order:0,role:"start_frame"}] else [] end),
      audio:false
    }')
  generation_id=$(curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary "${body}" "${BASE_URL}/api/videos/generate" | jq -er '.id')
  generation=$(poll_generation "${generation_id}")
  jq -e --arg quality "${quality}" '
    .requested_quality == $quality and .requested_aspect_ratio == "16:9" and
    (.requested_duration_seconds > 0) and (.requested_fps > 0) and
    (.requested_frames > 0) and (.inference_frames > 0) and
    (.actual_width > 0) and (.actual_height > 0) and
    (.actual_fps > 0) and (.actual_frames > 1) and (.actual_duration > 0) and
    ((.actual_duration - .requested_duration_seconds) < 0.15) and
    ((.actual_duration - .requested_duration_seconds) > -0.15) and
    (.actual_frames == ((.requested_duration_seconds * .requested_fps) | round))
  ' <<<"${generation}" >/dev/null
  printf '%s\n' "${generation}" > "${RESULT_ROOT}/${label}.generation.json"
  asset_id=$(jq -er '.output_asset_id' <<<"${generation}")
  validate_video "${label}" "${asset_id}" "${quality}"
}

# Ordre contractuel : une seule I2V 480p, inspection du plan réel, puis I2I.
run_video "i2v-480p" "IMAGE_TO_VIDEO" "IMAGE_TO_VIDEO" "480p" "${source_asset}"

generation_plan_line=$(compose logs --no-color worker | grep 'GENERATION_PLAN ' | tail -n 1)
[[ -n "${generation_plan_line}" ]] || { echo "GENERATION_PLAN absent des logs worker." >&2; exit 1; }
generation_plan_json=${generation_plan_line#*GENERATION_PLAN }
printf '%s\n' "${generation_plan_json}" | tee "${RESULT_ROOT}/generation-plan.json" \
  | jq -e '
      . as $plan |
      .capability == "IMAGE_TO_VIDEO" and
      (.inference_frames > 0) and
      (.strategy | type == "string") and
      (.vram_free > 0) and (.ram_available > 0) and
      (.forward_chunking | type == "boolean") and
      (($plan.decode_chunk_size == null) or ([1,2,4,8] | index($plan.decode_chunk_size)))
    ' >/dev/null

i2i_response=$(curl -fsS -X POST -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg model_id "${MODEL_ID}" --arg input_asset_id "${source_asset}" '{
    model_id:$model_id,
    mode:"IMAGE_TO_IMAGE",
    capability:"IMAGE_TO_IMAGE",
    prompt:"A restrained cinematic color grade",
    input_asset_id:$input_asset_id
  }')" "${BASE_URL}/api/images/generate")
i2i_generation_id=$(jq -er '.id' <<<"${i2i_response}")
i2i_terminal=""
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  i2i_terminal=$(curl -fsS "${BASE_URL}/api/generations/${i2i_generation_id}")
  i2i_status=$(jq -r '.status | ascii_downcase' <<<"${i2i_terminal}")
  if [[ "${i2i_status}" == "completed" || "${i2i_status}" == "failed" || "${i2i_status}" == "cancelled" ]]; then
    break
  fi
  sleep 3
done
printf '%s\n' "${i2i_terminal}" > "${RESULT_ROOT}/i2i.generation.json"
i2i_status=$(jq -r '.status | ascii_downcase' <<<"${i2i_terminal}")
if [[ "${i2i_status}" == "failed" ]]; then
  jq -e '(.error | type == "string" and length > 0 and (contains("La génération a échoué dans le runtime Diffusers.") | not))' \
    <<<"${i2i_terminal}" >/dev/null
  echo "GPU_I2I_STRUCTURED_FAILURE $(jq -r '.error' <<<"${i2i_terminal}")"
elif [[ "${i2i_status}" != "completed" ]]; then
  echo "I2I sans état terminal exploitable: ${i2i_status}" >&2
  exit 1
fi

jq -n \
  --arg version "${TARGET_VERSION}" \
  --arg model_id "${MODEL_ID}" \
  --arg gpu "${gpu_name}" \
  '{version:$version,model_id:$model_id,gpu:$gpu,result:"PASS"}' \
  > "${RESULT_ROOT}/summary.json"
echo "GPU_ACCEPTANCE_OK version=${TARGET_VERSION} model=${MODEL_ID} results=${RESULT_ROOT}"
