#!/usr/bin/env bash
set -Eeuo pipefail

NAME=${VIDIOAI_SERVER_NAME:-vidioai-gpu}
ZONE=${SCW_DEFAULT_ZONE:-fr-par-2}
CONFIRMATION=${1:-}
if [[ "${CONFIRMATION}" != "--confirm-destroy=${NAME}" ]]; then
  echo "Action destructive refusée. Utilisez --confirm-destroy=${NAME}" >&2
  exit 1
fi
SERVER_ID=$(scw instance server list zone="${ZONE}" name="${NAME}" -o json | jq -r '.[0].id // empty')
[[ -n "${SERVER_ID}" ]] || { echo "Serveur absent; rien à supprimer."; exit 0; }
scw instance server delete "${SERVER_ID}" zone="${ZONE}" with-ip=true with-block=true
echo "Serveur ${NAME} supprimé. Cette opération n'est pas récupérable."
