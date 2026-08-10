#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL=${1:-http://127.0.0.1:8080}
ALLOW_MISSING_MODES=${VIDIOAI_ACCEPT_ALLOW_MISSING_MODES:-false}
WORKDIR=$(mktemp -d /tmp/vidioai-gpu-acceptance.XXXXXX)
trap 'rm -rf "${WORKDIR}"' EXIT

SKIPPED_MODES=()

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Commande requise manquante: $1" >&2
    exit 1
  }
}

require_cmd curl
require_cmd jq
require_cmd ffprobe
require_cmd ffmpeg
require_cmd nvidia-smi

wait_ready() {
  local attempts=${1:-90}
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "${BASE_URL}/api/ready" | jq -e '.ready == true and .runtime == true and .gpu == true and .worker == true' >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

mode_for_capability() {
  local capability=${1:?capability requise}
  case "${capability}" in
    TEXT_TO_IMAGE)
      echo "TEXT_TO_IMAGE"
      ;;
    IMAGE_TO_IMAGE|INPAINTING|OUTPAINTING|IMAGE_VARIATION|IMAGE_UPSCALE|CONTROLLED_IMAGE_GENERATION)
      echo "IMAGE_TO_IMAGE"
      ;;
    TEXT_TO_VIDEO)
      echo "TEXT_TO_VIDEO"
      ;;
    IMAGE_TO_VIDEO|MULTI_IMAGE_TO_VIDEO|START_END_IMAGE_TO_VIDEO|KEYFRAMES_TO_VIDEO)
      echo "IMAGE_TO_VIDEO"
      ;;
    VIDEO_TO_VIDEO|VIDEO_INPAINTING|VIDEO_UPSCALE)
      echo "VIDEO_TO_VIDEO"
      ;;
    *)
      echo "Capacité non supportée: ${capability}" >&2
      return 1
      ;;
  esac
}

endpoint_for_capability() {
  local capability=${1:?capability requise}
  case "${capability}" in
    TEXT_TO_IMAGE|IMAGE_TO_IMAGE|INPAINTING|OUTPAINTING|IMAGE_VARIATION|IMAGE_UPSCALE|CONTROLLED_IMAGE_GENERATION)
      echo "/api/images/generate"
      ;;
    TEXT_TO_VIDEO|IMAGE_TO_VIDEO|MULTI_IMAGE_TO_VIDEO|START_END_IMAGE_TO_VIDEO|KEYFRAMES_TO_VIDEO|VIDEO_TO_VIDEO|VIDEO_INPAINTING|VIDEO_UPSCALE)
      echo "/api/videos/generate"
      ;;
    *)
      echo "Capacité non supportée: ${capability}" >&2
      return 1
      ;;
  esac
}

output_kind_for_capability() {
  local capability=${1:?capability requise}
  case "${capability}" in
    TEXT_TO_IMAGE|IMAGE_TO_IMAGE|INPAINTING|OUTPAINTING|IMAGE_VARIATION|IMAGE_UPSCALE|CONTROLLED_IMAGE_GENERATION)
      echo "image"
      ;;
    TEXT_TO_VIDEO|IMAGE_TO_VIDEO|MULTI_IMAGE_TO_VIDEO|START_END_IMAGE_TO_VIDEO|KEYFRAMES_TO_VIDEO|VIDEO_TO_VIDEO|VIDEO_INPAINTING|VIDEO_UPSCALE)
      echo "video"
      ;;
    *)
      echo "Capacité non supportée: ${capability}" >&2
      return 1
      ;;
  esac
}

select_model() {
  local mode=${1:?mode requis}
  curl -fsS --get \
    --data-urlencode "task=${mode}" \
    --data-urlencode "limit=80" \
    --data-urlencode "sort=compatibility" \
    "${BASE_URL}/api/models" \
    | jq -r '
      [.items[] | select(
        .engine_type == "ai" and
        .source_available == true and
        .runtime_supported == true and
        .installable == true and
        .gated == false and
        .private == false and
        .hardware_compatible == true
      )][0].id // empty
    '
}

install_or_load_model() {
  local model_id=${1:?model id requis}
  local model
  model=$(curl -fsS --get --data-urlencode "model_id=${model_id}" "${BASE_URL}/api/models/by-id")

  state=$(jq -r '.installation_state' <<<"${model}")
  if [[ "${state}" == "READY" ]]; then
    return 0
  fi

  if [[ "${state}" == "INSTALLED" ]]; then
    curl -fsS -X POST -H 'Content-Type: application/json' \
      --data-binary "$(jq -n --arg model_id "${model_id}" '{model_id:$model_id}')" \
      "${BASE_URL}/api/models/load" >/dev/null
  else
    revision=$(jq -r '.revision' <<<"${model}")
    job_id=$(curl -fsS -X POST -H 'Content-Type: application/json' \
      --data-binary "$(jq -n --arg model_id "${model_id}" --arg revision "${revision}" '{model_id:$model_id, revision:$revision}')" \
      "${BASE_URL}/api/models/install" | jq -er '.id')

    for ((i=1; i<=240; i++)); do
      job=$(curl -fsS "${BASE_URL}/api/jobs/${job_id}")
      status=$(jq -r '.status | ascii_downcase' <<<"${job}")
      if [[ "${status}" == "completed" ]]; then
        break
      fi
      if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
        jq . <<<"${job}" >&2
        return 1
      fi
      sleep 5
    done
  fi

  for ((i=1; i<=60; i++)); do
    model=$(curl -fsS --get --data-urlencode "model_id=${model_id}" "${BASE_URL}/api/models/by-id")
    if jq -e '.installation_state == "READY" and .runtime_ready == true' <<<"${model}" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

poll_generation() {
  local id=${1:?id requis}
  local attempts=${2:-180}
  for ((i=1; i<=attempts; i++)); do
    generation=$(curl -fsS "${BASE_URL}/api/generations/${id}")
    status=$(jq -r '.status | ascii_downcase' <<<"${generation}")
    if [[ "${status}" == "completed" ]]; then
      echo "${generation}"
      return 0
    fi
    if [[ "${status}" == "failed" || "${status}" == "cancelled" ]]; then
      jq . <<<"${generation}" >&2
      return 1
    fi
    sleep 5
  done
  return 1
}

upload_asset() {
  local path=${1:?fichier requis}
  local mime=${2:?mime requis}
  curl -fsS -X POST -F "file=@${path};type=${mime}" "${BASE_URL}/api/assets" | jq -er '.id'
}

make_inputs() {
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=red:s=256x256:d=1 -frames:v 1 "${WORKDIR}/start.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=blue:s=256x256:d=1 -frames:v 1 "${WORKDIR}/end.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=white:s=256x256:d=1 -frames:v 1 "${WORKDIR}/mask.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=green:s=256x256:d=1 -frames:v 1 "${WORKDIR}/control.png"
  ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc=size=256x256:rate=8 -t 2 -pix_fmt yuv420p "${WORKDIR}/source.mp4"
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

echo "[gpu-acceptance] 1) Vérification NVIDIA"
nvidia-smi >/dev/null

if ! wait_ready 120; then
  echo "Backend /ready n'est pas prêt (runtime/GPU/worker)." >&2
  curl -fsS "${BASE_URL}/api/ready" | jq . >&2 || true
  exit 1
fi

make_inputs
start_asset=$(upload_asset "${WORKDIR}/start.png" "image/png")
end_asset=$(upload_asset "${WORKDIR}/end.png" "image/png")
mask_asset=$(upload_asset "${WORKDIR}/mask.png" "image/png")
control_asset=$(upload_asset "${WORKDIR}/control.png" "image/png")
video_asset=$(upload_asset "${WORKDIR}/source.mp4" "video/mp4")

run_capability() {
  local capability=${1:?capabilité requise}
  local mode endpoint output_kind
  mode=$(mode_for_capability "${capability}")
  endpoint=$(endpoint_for_capability "${capability}")
  output_kind=$(output_kind_for_capability "${capability}")

  model_id=$(select_model "${mode}")
  if [[ -z "${model_id}" ]]; then
    echo "[gpu-acceptance] ${capability}: aucun modèle public raisonnable disponible (mode ${mode}, signalé explicitement)."
    SKIPPED_MODES+=("${capability}")
    return 0
  fi

  echo "[gpu-acceptance] ${capability}: modèle ${model_id}"
  install_or_load_model "${model_id}" || {
    echo "Installation/chargement impossible pour ${model_id} (${capability})" >&2
    return 1
  }

  local input_asset_id=""
  local mask_id=""
  local control_id=""
  local input_images_json='[]'

  case "${capability}" in
    IMAGE_TO_IMAGE|IMAGE_VARIATION|IMAGE_UPSCALE|IMAGE_TO_VIDEO)
      input_asset_id="${start_asset}"
      ;;
    INPAINTING|OUTPAINTING)
      input_asset_id="${start_asset}"
      mask_id="${mask_asset}"
      ;;
    CONTROLLED_IMAGE_GENERATION)
      input_asset_id="${start_asset}"
      control_id="${control_asset}"
      ;;
    MULTI_IMAGE_TO_VIDEO)
      input_asset_id="${start_asset}"
      input_images_json=$(jq -n --arg start "${start_asset}" --arg end "${end_asset}" '[{asset_id:$start,order:0,role:"start_frame"},{asset_id:$end,order:1,role:"reference"}]')
      ;;
    START_END_IMAGE_TO_VIDEO)
      input_asset_id="${start_asset}"
      input_images_json=$(jq -n --arg start "${start_asset}" --arg end "${end_asset}" '[{asset_id:$start,order:0,role:"start_frame"},{asset_id:$end,order:1,role:"end_frame"}]')
      ;;
    KEYFRAMES_TO_VIDEO)
      input_asset_id="${start_asset}"
      input_images_json=$(jq -n --arg start "${start_asset}" --arg end "${end_asset}" '[{asset_id:$start,order:0,role:"keyframe"},{asset_id:$end,order:1,role:"keyframe"}]')
      ;;
    VIDEO_TO_VIDEO|VIDEO_INPAINTING|VIDEO_UPSCALE)
      input_asset_id="${video_asset}"
      ;;
  esac

  payload=$(jq -n \
    --arg mode "${mode}" \
    --arg capability "${capability}" \
    --arg model_id "${model_id}" \
    --arg prompt "VidioAI GPU acceptance test ${capability}" \
    --arg input_asset_id "${input_asset_id}" \
    --arg mask_asset_id "${mask_id}" \
    --arg control_asset_id "${control_id}" \
    --argjson input_images "${input_images_json}" \
    '{
      mode:$mode,
      capability:$capability,
      prompt:$prompt,
      model_id:$model_id,
      input_asset_id:(if ($input_asset_id|length) > 0 then $input_asset_id else null end),
      mask_asset_id:(if ($mask_asset_id|length) > 0 then $mask_asset_id else null end),
      control_asset_id:(if ($control_asset_id|length) > 0 then $control_asset_id else null end),
      input_images:$input_images,
      duration_seconds:2,
      resolution:"720p"
    }')

  generation_id=$(curl -fsS -X POST -H 'Content-Type: application/json' --data-binary "${payload}" "${BASE_URL}${endpoint}" | jq -er '.id')
  result=$(poll_generation "${generation_id}") || return 1
  output_asset_id=$(jq -er '.output_asset_id' <<<"${result}")

  if [[ "${output_kind}" == "image" ]]; then
    out_file="${WORKDIR}/${capability}.png"
    curl -fsS "${BASE_URL}/api/assets/${output_asset_id}" -o "${out_file}"
    validate_png "${out_file}" || {
      echo "Sortie ${capability} invalide (PNG non valide)." >&2
      return 1
    }
  else
    out_file="${WORKDIR}/${capability}.mp4"
    curl -fsS "${BASE_URL}/api/assets/${output_asset_id}" -o "${out_file}"
    validate_video "${out_file}" || {
      echo "Sortie ${capability} invalide (vidéo non lisible)." >&2
      return 1
    }
  fi

  echo "[gpu-acceptance] ${capability}: OK"
}

for capability in \
  TEXT_TO_IMAGE \
  IMAGE_TO_IMAGE \
  INPAINTING \
  OUTPAINTING \
  IMAGE_VARIATION \
  IMAGE_UPSCALE \
  CONTROLLED_IMAGE_GENERATION \
  TEXT_TO_VIDEO \
  IMAGE_TO_VIDEO \
  MULTI_IMAGE_TO_VIDEO \
  START_END_IMAGE_TO_VIDEO \
  KEYFRAMES_TO_VIDEO \
  VIDEO_TO_VIDEO \
  VIDEO_INPAINTING \
  VIDEO_UPSCALE
do
  run_capability "${capability}"
done

curl -fsS "${BASE_URL}/api/queue" | jq -e '.active >= 0 and .queued >= 0 and .completed >= 0' >/dev/null
curl -fsS "${BASE_URL}/api/resources" | jq -e '.queue_total >= 0' >/dev/null

if (( ${#SKIPPED_MODES[@]} > 0 )); then
  echo "[gpu-acceptance] Capacités non validées automatiquement: ${SKIPPED_MODES[*]}"
  if [[ "${ALLOW_MISSING_MODES}" != "true" ]]; then
    echo "Échec bloquant: certaines capacités n'ont pas de modèle public raisonnable sélectionnable." >&2
    exit 3
  fi
fi

echo "GPU ACCEPTANCE OK"
