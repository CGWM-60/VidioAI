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
curl -fsS "${BASE_URL}/api/models" | jq -e 'length >= 8' >/dev/null
curl -fsS "${BASE_URL}/api/system" | jq -e '.source and .system and .cpu and .ram and .gpus and (.storage.volumes | type == "array")' >/dev/null
curl -fsS "${BASE_URL}/api/dashboard" | jq -e '.generations_total >= 0' >/dev/null
curl -fsS "${BASE_URL}/" | grep -q '<html'

# Validation matérielle complète facultative mais recommandée lors du premier
# déploiement. Elle télécharge/valide/charge le modèle puis produit un vrai PNG.
if [[ -n "${VIDIOAI_SMOKE_AI_MODEL_ID:-}" ]]; then
  MODEL_ID=${VIDIOAI_SMOKE_AI_MODEL_ID}
  MODEL=$(curl -fsS "${BASE_URL}/api/models/${MODEL_ID}")
  if ! jq -e '.installation_state == "READY"' <<<"${MODEL}" >/dev/null; then
    JOB_ID=$(curl -fsS -X POST "${BASE_URL}/api/models/${MODEL_ID}/install" | jq -r '.id')
    for ((attempt=1; attempt<=${SMOKE_MODEL_ATTEMPTS:-180}; attempt++)); do
      JOB=$(curl -fsS "${BASE_URL}/api/jobs/${JOB_ID}")
      STATUS=$(jq -r '.status' <<<"${JOB}")
      [[ "${STATUS}" == "completed" ]] && break
      [[ "${STATUS}" == "failed" ]] && { jq . <<<"${JOB}" >&2; exit 1; }
      sleep 10
    done
  fi
  GENERATION_ID=$(curl -fsS -X POST -H 'Content-Type: application/json' \
    -d "{\"mode\":\"TEXT_TO_IMAGE\",\"prompt\":\"A small violet cube on a neutral studio background\",\"model_id\":\"${MODEL_ID}\"}" \
    "${BASE_URL}/api/images/generate" | jq -r '.id')
  for ((attempt=1; attempt<=${SMOKE_GENERATION_ATTEMPTS:-120}; attempt++)); do
    GENERATION=$(curl -fsS "${BASE_URL}/api/generations/${GENERATION_ID}")
    STATUS=$(jq -r '.status' <<<"${GENERATION}")
    if [[ "${STATUS}" == "COMPLETED" ]]; then
      ASSET_ID=$(jq -r '.output_asset_id' <<<"${GENERATION}")
      curl -fsS "${BASE_URL}/api/assets/${ASSET_ID}" -o /tmp/vidioai-smoke.png
      file /tmp/vidioai-smoke.png | grep -q PNG
      break
    fi
    [[ "${STATUS}" == "FAILED" ]] && { jq . <<<"${GENERATION}" >&2; exit 1; }
    sleep 5
  done
fi
echo "Smoke tests réussis sur ${BASE_URL}."
