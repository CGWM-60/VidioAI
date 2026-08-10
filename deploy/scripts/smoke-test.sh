#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL=${1:-http://127.0.0.1}
PROFILE=${VIDIOAI_SMOKE_PROFILE:-GPU_PRODUCTION}
ATTEMPTS=${SMOKE_ATTEMPTS:-30}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Commande requise manquante: $1" >&2
    exit 1
  }
}

require_cmd curl
require_cmd jq
require_cmd ffmpeg
require_cmd ffprobe

poll_generation() {
  local id=${1:?id requis}
  local attempts=${2:-120}
  local allow_failed=${3:-false}
  for ((i=1; i<=attempts; i++)); do
    generation=$(curl -fsS "${BASE_URL}/api/generations/${id}")
    status=$(jq -r '.status | ascii_downcase' <<<"${generation}")
    if [[ "${status}" == "completed" ]]; then
      echo "${generation}"
      return 0
    fi
    if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      if [[ "${allow_failed}" == "true" ]]; then
        echo "${generation}"
        return 0
      fi
      jq . <<<"${generation}" >&2
      return 1
    fi
    sleep 3
  done
  return 1
}

validate_png() {
  local file=${1:?file requis}
  file "${file}" | grep -q 'PNG image data'
  od -An -tx1 -N8 "${file}" | tr -d ' \n' | grep -qi '^89504e470d0a1a0a$'
}

validate_video() {
  local file=${1:?file requis}
  ffprobe -v error -show_streams -show_format "${file}" >/dev/null
}

for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  if curl -fsS "${BASE_URL}/api/ready" | jq -e '.ready == true' >/dev/null; then break; fi
  if [[ "${attempt}" -eq "${ATTEMPTS}" ]]; then echo "Readiness en échec." >&2; exit 1; fi
  sleep 2
done

curl -fsS "${BASE_URL}/healthcheck" >/dev/null
curl -fsS "${BASE_URL}/api/health" | jq -e '.status == "ok"' >/dev/null
READY=$(curl -fsS "${BASE_URL}/api/ready")
echo "${READY}" | jq -e '.storage_writable and .scratch_writable and .ffmpeg and .queue and .s3' >/dev/null
if [[ "${PROFILE}" == "GPU_PRODUCTION" ]]; then
  echo "${READY}" | jq -e '.profile == "GPU_PRODUCTION" and .host_agent and (.system_source == "host") and .worker and .runtime and .gpu' >/dev/null
fi
curl -fsS "${BASE_URL}/api/resources" | jq -e '.system.source and .system.cpu.source and .system.ram.source and .system.storage.source and (.queue_total >= 0)' >/dev/null
MODELS=$(curl -fsS "${BASE_URL}/api/models?limit=80&sort=compatibility")
jq -e '.items | type == "array" and length > 0' <<<"${MODELS}" >/dev/null
curl -fsS "${BASE_URL}/api/system" | jq -e '.source and .system and .cpu and .ram and .gpus and (.storage.volumes | type == "array")' >/dev/null
curl -fsS "${BASE_URL}/api/dashboard" | jq -e '.generations_total >= 0' >/dev/null
curl -fsS "${BASE_URL}/" | grep -q '<html'

if [[ -n "${VIDIOAI_WORKER_URL:-}" ]]; then
  curl -fsS "${VIDIOAI_WORKER_URL}/health" | jq -e '.status == "ok" and .service == "vidioai-gpu-worker"' >/dev/null
  if [[ -n "${VIDIOAI_WORKER_TOKEN:-}" ]]; then
    curl -fsS -H "X-VidioAI-Worker-Token: ${VIDIOAI_WORKER_TOKEN}" "${VIDIOAI_WORKER_URL}/ready" | jq -e '.ready == true or .ready == false' >/dev/null
  fi
fi

if [[ "${PROFILE}" == "LOCAL" ]]; then
  WORKDIR=$(mktemp -d /tmp/vidioai-local-smoke.XXXXXX)
  trap 'rm -rf "${WORKDIR}"' EXIT
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=red:s=256x256:d=1 -frames:v 1 "${WORKDIR}/start.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=green:s=256x256:d=1 -frames:v 1 "${WORKDIR}/end.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=white:s=256x256:d=1 -frames:v 1 "${WORKDIR}/mask.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=blue:s=256x256:d=1 -frames:v 1 "${WORKDIR}/control.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc=size=256x256:rate=8 -t 2 -pix_fmt yuv420p "${WORKDIR}/source.mp4"

  upload_asset() {
    local path=${1:?file requis}
    local mime=${2:?mime requis}
    curl -fsS -X POST -F "file=@${path};type=${mime}" "${BASE_URL}/api/assets" | jq -er '.id'
  }

  start_asset=$(upload_asset "${WORKDIR}/start.png" "image/png")
  end_asset=$(upload_asset "${WORKDIR}/end.png" "image/png")
  mask_asset=$(upload_asset "${WORKDIR}/mask.png" "image/png")
  control_asset=$(upload_asset "${WORKDIR}/control.png" "image/png")
  video_asset=$(upload_asset "${WORKDIR}/source.mp4" "video/mp4")

  run_image_contract() {
    local capability=${1:?capability requise}
    local input_asset_id=${2:-}
    local mask=${3:-}
    local control=${4:-}
    payload=$(jq -n \
      --arg mode "$([[ "${capability}" == "TEXT_TO_IMAGE" ]] && echo "TEXT_TO_IMAGE" || echo "IMAGE_TO_IMAGE")" \
      --arg capability "${capability}" \
      --arg prompt "VidioAI non-GPU smoke ${capability}" \
      --arg model_id "vidio-canvas-local" \
      --arg input_asset_id "${input_asset_id}" \
      --arg mask_asset_id "${mask}" \
      --arg control_asset_id "${control}" \
      '{
        mode:$mode,
        capability:$capability,
        prompt:$prompt,
        model_id:$model_id,
        input_asset_id:(if ($input_asset_id|length) > 0 then $input_asset_id else null end),
        mask_asset_id:(if ($mask_asset_id|length) > 0 then $mask_asset_id else null end),
        control_asset_id:(if ($control_asset_id|length) > 0 then $control_asset_id else null end)
      }')
    generation_id=$(curl -fsS -X POST -H 'Content-Type: application/json' --data-binary "${payload}" "${BASE_URL}/api/images/generate" | jq -er '.id')
    result=$(poll_generation "${generation_id}")
    asset_id=$(jq -er '.output_asset_id' <<<"${result}")
    out_file="${WORKDIR}/${capability}.png"
    curl -fsS "${BASE_URL}/api/assets/${asset_id}" -o "${out_file}"
    validate_png "${out_file}"
  }

  run_video_contract() {
    local capability=${1:?capability requise}
    local mode=${2:?mode requis}
    local input_asset_id=${3:-}
    local input_images_json=${4:-[]}
    payload=$(jq -n \
      --arg mode "${mode}" \
      --arg capability "${capability}" \
      --arg prompt "VidioAI non-GPU smoke ${capability}" \
      --arg model_id "vidio-motion-local" \
      --arg input_asset_id "${input_asset_id}" \
      --argjson input_images "${input_images_json}" \
      '{
        mode:$mode,
        capability:$capability,
        prompt:$prompt,
        model_id:$model_id,
        input_asset_id:(if ($input_asset_id|length) > 0 then $input_asset_id else null end),
        input_images:$input_images,
        duration_seconds:2,
        resolution:"720p"
      }')
    generation_id=$(curl -fsS -X POST -H 'Content-Type: application/json' --data-binary "${payload}" "${BASE_URL}/api/videos/generate" | jq -er '.id')
    result=$(poll_generation "${generation_id}" 120 true)
    status=$(jq -r '.status | ascii_downcase' <<<"${result}")
    if [[ "${status}" == "completed" ]]; then
      asset_id=$(jq -er '.output_asset_id' <<<"${result}")
      out_file="${WORKDIR}/${capability}.mp4"
      curl -fsS "${BASE_URL}/api/assets/${asset_id}" -o "${out_file}"
      validate_video "${out_file}"
      return 0
    fi

    # En non-GPU pur, les capacités vidéo peuvent échouer proprement faute de
    # worker GPU. Ce cas reste conforme au contrat API si l'erreur est claire.
    error_message=$(jq -r '.error // ""' <<<"${result}")
    if [[ -z "${error_message}" ]]; then
      echo "${result}" >&2
      echo "Échec ${capability} sans message d'erreur explicite." >&2
      return 1
    fi
    if [[ ! "${error_message}" =~ [Ww]orker|[Gg][Pp][Uu]|runtime ]]; then
      echo "${result}" >&2
      echo "Échec ${capability} inattendu en LOCAL non-GPU: ${error_message}" >&2
      return 1
    fi
    echo "[local-smoke] ${capability}: échec attendu sans GPU (${error_message})"
  }

  run_image_contract "TEXT_TO_IMAGE"
  run_image_contract "IMAGE_TO_IMAGE" "${start_asset}"
  run_image_contract "INPAINTING" "${start_asset}" "${mask_asset}"
  run_image_contract "OUTPAINTING" "${start_asset}" "${mask_asset}"
  run_image_contract "IMAGE_VARIATION" "${start_asset}"
  run_image_contract "IMAGE_UPSCALE" "${start_asset}"
  run_image_contract "CONTROLLED_IMAGE_GENERATION" "${start_asset}" "" "${control_asset}"

  run_video_contract "TEXT_TO_VIDEO" "TEXT_TO_VIDEO"
  run_video_contract "IMAGE_TO_VIDEO" "IMAGE_TO_VIDEO" "${start_asset}"
  run_video_contract "MULTI_IMAGE_TO_VIDEO" "IMAGE_TO_VIDEO" "${start_asset}" "$(jq -n --arg start "${start_asset}" --arg end "${end_asset}" '[{asset_id:$start,order:0,role:"start_frame"},{asset_id:$end,order:1,role:"reference"}]')"
  run_video_contract "START_END_IMAGE_TO_VIDEO" "IMAGE_TO_VIDEO" "${start_asset}" "$(jq -n --arg start "${start_asset}" --arg end "${end_asset}" '[{asset_id:$start,order:0,role:"start_frame"},{asset_id:$end,order:1,role:"end_frame"}]')"
  run_video_contract "KEYFRAMES_TO_VIDEO" "IMAGE_TO_VIDEO" "${start_asset}" "$(jq -n --arg start "${start_asset}" --arg end "${end_asset}" '[{asset_id:$start,order:0,role:"keyframe"},{asset_id:$end,order:1,role:"keyframe"}]')"
  run_video_contract "VIDEO_TO_VIDEO" "VIDEO_TO_VIDEO" "${video_asset}"
  run_video_contract "VIDEO_INPAINTING" "VIDEO_TO_VIDEO" "${video_asset}"
  run_video_contract "VIDEO_UPSCALE" "VIDEO_TO_VIDEO" "${video_asset}"

  echo "Smoke LOCAL: matrice des 15 capacités validée (contrat API non-GPU)."
fi

# En GPU_PRODUCTION, seule une vraie inférence IA est acceptée.
if [[ "${PROFILE}" == "GPU_PRODUCTION" ]]; then
  MODEL_ID=${VIDIOAI_SMOKE_AI_MODEL_ID:-}
  if [[ -z "${MODEL_ID}" ]]; then
    MODEL_ID=$(jq -r '
      [.items[] | select(
        .kind == "IMAGE" and .engine_type == "ai" and
        .runtime_supported == true and .hardware_compatible == true and
        .source_available == true and .installable == true and
        .gated == false and .private == false
      )][0].id // empty
    ' <<<"${MODELS}")
  fi
  if [[ -z "${MODEL_ID}" ]]; then
    echo "Aucun modèle Hugging Face IMAGE réellement installable n'est disponible pour le smoke test." >&2
    exit 1
  fi
  MODEL=$(curl -fsS --get --data-urlencode "model_id=${MODEL_ID}" "${BASE_URL}/api/models/by-id")
  if ! jq -e '.runtime_supported and .hardware_compatible and .source_available' <<<"${MODEL}" >/dev/null; then
    jq '.compatibility_checks' <<<"${MODEL}" >&2
    exit 1
  fi
  GENERATION_BODY=$(jq -n --arg model_id "${MODEL_ID}" '{mode:"TEXT_TO_IMAGE",prompt:"A small violet cube on a neutral studio background",model_id:$model_id}')
  GENERATION_ID=$(curl -fsS -X POST -H 'Content-Type: application/json' --data-binary "${GENERATION_BODY}" "${BASE_URL}/api/images/generate" | jq -er '.id')
  GENERATION=$(poll_generation "${GENERATION_ID}" "${SMOKE_GENERATION_ATTEMPTS:-120}")
  ASSET_ID=$(jq -er '.output_asset_id' <<<"${GENERATION}")
  SMOKE_PNG=$(mktemp /tmp/vidioai-smoke.XXXXXX.png)
  trap 'rm -f "${SMOKE_PNG}"' EXIT
  curl -fsS "${BASE_URL}/api/assets/${ASSET_ID}" -o "${SMOKE_PNG}"
  validate_png "${SMOKE_PNG}"
fi

echo "Smoke tests réussis sur ${BASE_URL}."
