#!/usr/bin/env bash
set -Eeuo pipefail

# Synchronisation L3 optionnelle avec n'importe quel endpoint S3. Les secrets
# restent fournis par l'environnement et ne sont jamais écrits dans les logs.
: "${AWS_S3_BUCKET:?AWS_S3_BUCKET requis}"
: "${AWS_ENDPOINT_URL_S3:?AWS_ENDPOINT_URL_S3 requis}"
MODE=${1:-push}
SOURCE_OUTPUTS=${VIDIOAI_OUTPUTS_DIR:-/var/lib/vidioai/outputs}
SOURCE_MODELS=${VIDIOAI_MODELS_DIR:-/var/lib/vidioai/scratch/models}

command -v aws >/dev/null 2>&1 || { echo "aws-cli est requis." >&2; exit 1; }
if [[ "${MODE}" == "push" ]]; then
  aws s3 sync "${SOURCE_OUTPUTS}" "s3://${AWS_S3_BUCKET}/outputs" --endpoint-url "${AWS_ENDPOINT_URL_S3}" --only-show-errors
  aws s3 sync "${SOURCE_MODELS}" "s3://${AWS_S3_BUCKET}/models" --endpoint-url "${AWS_ENDPOINT_URL_S3}" --only-show-errors
elif [[ "${MODE}" == "pull" ]]; then
  aws s3 sync "s3://${AWS_S3_BUCKET}/models" "${SOURCE_MODELS}" --endpoint-url "${AWS_ENDPOINT_URL_S3}" --only-show-errors
else
  echo "Mode attendu: push ou pull" >&2; exit 1
fi
