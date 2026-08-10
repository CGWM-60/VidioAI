#!/usr/bin/env bash
set -Eeuo pipefail

# Création volontairement explicite : aucun défaut ne choisit silencieusement
# une machine GPU coûteuse. Le CLI Scaleway doit déjà être authentifié.
: "${SCW_DEFAULT_PROJECT_ID:?SCW_DEFAULT_PROJECT_ID requis}"
NAME=${VIDIOAI_SERVER_NAME:-vidioai-gpu}
TYPE=${VIDIOAI_SERVER_TYPE:?Définissez VIDIOAI_SERVER_TYPE (ex. L4-1-24G)}
ZONE=${SCW_DEFAULT_ZONE:-fr-par-2}
IMAGE=${VIDIOAI_SERVER_IMAGE:-ubuntu_jammy}

command -v scw >/dev/null 2>&1 || { echo "Installez le CLI Scaleway." >&2; exit 1; }
if scw instance server list zone="${ZONE}" name="${NAME}" -o json | jq -e 'length > 0' >/dev/null; then
  echo "Le serveur ${NAME} existe déjà; aucune création effectuée."
  exit 0
fi
scw instance server create name="${NAME}" type="${TYPE}" image="${IMAGE}" zone="${ZONE}" project-id="${SCW_DEFAULT_PROJECT_ID}" ip=new start=true
scw instance server list zone="${ZONE}" name="${NAME}"
