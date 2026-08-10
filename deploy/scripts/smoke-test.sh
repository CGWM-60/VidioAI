#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL=${1:-http://127.0.0.1}
ATTEMPTS=${SMOKE_ATTEMPTS:-30}

# La boucle couvre le temps de warm-up sans transformer un simple démarrage lent
# en échec définitif. La réponse finale est ensuite validée structurellement.
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  if curl -fsS "${BASE_URL}/api/ready" | jq -e '.ready == true' >/dev/null; then break; fi
  if [[ "${attempt}" -eq "${ATTEMPTS}" ]]; then echo "Readiness en échec." >&2; exit 1; fi
  sleep 2
done

curl -fsS "${BASE_URL}/healthcheck" >/dev/null
curl -fsS "${BASE_URL}/api/health" | jq -e '.status == "ok"' >/dev/null
READY=$(curl -fsS "${BASE_URL}/api/ready")
echo "${READY}" | jq -e '.storage_writable and .scratch_writable and .ffmpeg and .queue and .s3' >/dev/null
if [[ "${VIDIOAI_SMOKE_PROFILE:-GPU_PRODUCTION}" == "GPU_PRODUCTION" ]]; then
  echo "${READY}" | jq -e '.profile == "GPU_PRODUCTION" and .host_agent and (.system_source == "host") and .worker and .runtime and .gpu' >/dev/null
fi
curl -fsS "${BASE_URL}/api/resources" | jq -e '.system.source and .system.cpu.source and .system.ram.source and .system.storage.source and (.queue_total >= 0)' >/dev/null
MODELS=$(curl -fsS "${BASE_URL}/api/models?limit=60&sort=compatibility")
jq -e '.items | type == "array" and length > 0' <<<"${MODELS}" >/dev/null
curl -fsS "${BASE_URL}/api/system" | jq -e '.source and .system and .cpu and .ram and .gpus and (.storage.volumes | type == "array")' >/dev/null
curl -fsS "${BASE_URL}/api/dashboard" | jq -e '.generations_total >= 0' >/dev/null
curl -fsS "${BASE_URL}/" | grep -q '<html'

# En GPU_PRODUCTION la validation IA n'est jamais facultative : le test choisit
# un vrai repository T2I dynamique, ou respecte l'identifiant explicite fourni.
if [[ "${VIDIOAI_SMOKE_PROFILE:-GPU_PRODUCTION}" == "GPU_PRODUCTION" ]]; then
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
  echo "Smoke GPU avec le repository Hugging Face: ${MODEL_ID}"

  # --get/--data-urlencode préservent exactement le slash organisation/modèle.
  MODEL=$(curl -fsS --get --data-urlencode "model_id=${MODEL_ID}" "${BASE_URL}/api/models/by-id")
  if ! jq -e '.runtime_supported and .hardware_compatible and .source_available' <<<"${MODEL}" >/dev/null; then
    jq '.compatibility_checks' <<<"${MODEL}" >&2
    exit 1
  fi

  INSTALLATION_STATE=$(jq -r '.installation_state' <<<"${MODEL}")
  if [[ "${INSTALLATION_STATE}" == "INSTALLED" ]]; then
    # Un snapshot valide mais déchargé est remis en VRAM sans téléchargement.
    jq -n --arg model_id "${MODEL_ID}" '{model_id:$model_id}' \
      | curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @- \
          "${BASE_URL}/api/models/load" >/dev/null
  elif [[ "${INSTALLATION_STATE}" != "READY" ]]; then
    INSTALL_BODY=$(jq -n --arg model_id "${MODEL_ID}" --arg revision "$(jq -r '.revision' <<<"${MODEL}")" \
      '{model_id:$model_id, revision:$revision}')
    JOB_ID=$(curl -fsS -X POST -H 'Content-Type: application/json' \
      --data-binary "${INSTALL_BODY}" "${BASE_URL}/api/models/install" | jq -er '.id')
    INSTALL_COMPLETE=false
    for ((attempt=1; attempt<=${SMOKE_MODEL_ATTEMPTS:-180}; attempt++)); do
      JOB=$(curl -fsS "${BASE_URL}/api/jobs/${JOB_ID}")
      STATUS=$(jq -r '.status | ascii_downcase' <<<"${JOB}")
      if [[ "${STATUS}" == "completed" ]]; then INSTALL_COMPLETE=true; break; fi
      if [[ "${STATUS}" == "failed" || "${STATUS}" == "cancelled" ]]; then jq . <<<"${JOB}" >&2; exit 1; fi
      sleep 10
    done
    [[ "${INSTALL_COMPLETE}" == true ]] || { echo "Timeout d'installation du modèle ${MODEL_ID}." >&2; exit 1; }
  fi

  MODEL_READY=false
  for ((attempt=1; attempt<=${SMOKE_READY_ATTEMPTS:-30}; attempt++)); do
    MODEL=$(curl -fsS --get --data-urlencode "model_id=${MODEL_ID}" "${BASE_URL}/api/models/by-id")
    if jq -e '.installation_state == "READY" and .runtime_ready == true' <<<"${MODEL}" >/dev/null; then MODEL_READY=true; break; fi
    sleep 2
  done
  [[ "${MODEL_READY}" == true ]] || { echo "Le modèle ${MODEL_ID} n'a jamais atteint READY." >&2; exit 1; }

  GENERATION_BODY=$(jq -n --arg model_id "${MODEL_ID}" \
    '{mode:"TEXT_TO_IMAGE",prompt:"A small violet cube on a neutral studio background",model_id:$model_id}')
  GENERATION_ID=$(curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary "${GENERATION_BODY}" "${BASE_URL}/api/images/generate" | jq -er '.id')
  GENERATION_COMPLETE=false
  for ((attempt=1; attempt<=${SMOKE_GENERATION_ATTEMPTS:-120}; attempt++)); do
    GENERATION=$(curl -fsS "${BASE_URL}/api/generations/${GENERATION_ID}")
    STATUS=$(jq -r '.status | ascii_downcase' <<<"${GENERATION}")
    if [[ "${STATUS}" == "completed" ]]; then GENERATION_COMPLETE=true; break; fi
    if [[ "${STATUS}" == "failed" || "${STATUS}" == "cancelled" ]]; then jq . <<<"${GENERATION}" >&2; exit 1; fi
    sleep 5
  done
  [[ "${GENERATION_COMPLETE}" == true ]] || { echo "Timeout de génération pour ${GENERATION_ID}." >&2; exit 1; }

  ASSET_ID=$(jq -er '.output_asset_id' <<<"${GENERATION}")
  SMOKE_PNG=$(mktemp /tmp/vidioai-smoke.XXXXXX.png)
  trap 'rm -f "${SMOKE_PNG}"' EXIT
  curl -fsS "${BASE_URL}/api/assets/${ASSET_ID}" -o "${SMOKE_PNG}"
  file "${SMOKE_PNG}" | grep -q 'PNG image data'
  # La signature binaire protège contre une page JSON/HTML enregistrée en .png.
  od -An -tx1 -N8 "${SMOKE_PNG}" | tr -d ' \n' | grep -qi '^89504e470d0a1a0a$'
  echo "PNG réel validé: génération ${GENERATION_ID}, asset ${ASSET_ID}."
fi
echo "Smoke tests réussis sur ${BASE_URL}."
