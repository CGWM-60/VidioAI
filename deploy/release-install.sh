#!/usr/bin/env bash
set -Eeuo pipefail

# Installation d'une release déjà publiée. Le build et la publication restent
# la responsabilité de scripts/build-release.sh : la machine GPU ne reconstruit
# jamais silencieusement une image immuable.
SOURCE_PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERSION=${1:-}
REQUESTED_VERSION=${VERSION}
INSTALL_DIR=${VIDIOAI_INSTALL_DIR:-${VIDIOAI_PROJECT_DIR:-/opt/vidioai}}
ENV_FILE=${VIDIOAI_ENV_FILE:-${INSTALL_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${INSTALL_DIR}/compose.production.yml}
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/vidioai-release-install.XXXXXX")
ARCHIVE_FILE="${WORK_DIR}/deployment.tar.gz"
MANIFEST_FILE="${WORK_DIR}/release.json"
EXTRACT_DIR="${WORK_DIR}/extract"
CURRENT_STEP=1
CURRENT_STAGE=Version
DEPLOY_SUCCEEDED=false

finish() {
  local status=$?
  trap - EXIT
  rm -rf "${WORK_DIR}"
  if [[ "${DEPLOY_SUCCEEDED}" != "true" ]]; then
    if [[ ${status} -eq 0 ]]; then
      status=1
    fi
    printf 'FAILED VidioAI %s at [%s/7] %s\n' "${VERSION:-<missing>}" "${CURRENT_STEP}" "${CURRENT_STAGE}" >&2
  fi
  exit "${status}"
}
trap finish EXIT

stage() {
  CURRENT_STEP=${1:?numero_etape_requis}
  CURRENT_STAGE=${2:?nom_etape_requis}
  printf '[%s/7] %s\n' "${CURRENT_STEP}" "${CURRENT_STAGE}"
}

fail() {
  printf '%s\n' "$*" >&2
  return 1
}

expand_version() {
  local value=${1:?valeur requise}
  printf '%s\n' "${value//\{version\}/${VERSION}}"
}

upsert_env() {
  local key=${1:?clé requise}
  local value=${2-}
  local temporary="${ENV_FILE}.tmp.$$"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    awk -v key="${key}" -v value="${value}" '
      index($0, key "=") == 1 { print key "=" value; next }
      { print }
    ' "${ENV_FILE}" > "${temporary}"
  else
    cp "${ENV_FILE}" "${temporary}"
    printf '%s=%s\n' "${key}" "${value}" >> "${temporary}"
  fi
  mv "${temporary}" "${ENV_FILE}"
}

read_release_manifest_from_archive() {
  local archive=${1:?archive requise}
  tar -xOzf "${archive}" "release-${VERSION}/release.json" > "${MANIFEST_FILE}" \
    || fail "Manifest release-${VERSION}/release.json absent de l'archive."
}

validate_manifest_and_remote_images() {
  jq -e --arg version "${VERSION}" '
    .version == $version
    and (.registry | type == "string" and length > 0)
    and ([.images.backend, .images.frontend, .images.worker]
      | all(type == "string" and test("^sha256:[0-9a-f]{64}$")))
  ' "${MANIFEST_FILE}" >/dev/null \
    || fail "Manifest de release invalide ou version contradictoire."

  command -v docker >/dev/null 2>&1 \
    || fail "Docker/buildx est requis pour vérifier les tags immuables avant installation."

  local registry service reference expected actual
  registry=$(jq -r '.registry' "${MANIFEST_FILE}")
  for service in backend frontend worker; do
    reference="${registry}/${service}:${VERSION}"
    expected=$(jq -r --arg service "${service}" '.images[$service]' "${MANIFEST_FILE}")
    actual=$(docker buildx imagetools inspect "${reference}" --format '{{json .Manifest.Digest}}' 2>/dev/null | tr -d '"') \
      || fail "Image distante introuvable avant installation: ${reference}"
    [[ "${actual}" == "${expected}" ]] \
      || fail "Digest distant contradictoire pour ${reference}: ${actual:-absent} != ${expected}"
  done
}

stage 1 Version
[[ -n "${VERSION}" ]] || fail "Usage: ./deploy/release-install.sh <version-immuable>"
[[ "${VERSION}" != "latest" && "${VERSION}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]+$ ]] \
  || fail "Version immuable invalide: ${VERSION}"

# La configuration déjà installée fournit les credentials de lecture S3, sans
# pouvoir remplacer la version explicitement demandée sur la ligne de commande.
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
VERSION=${REQUESTED_VERSION}
export VIDIOAI_VERSION="${VERSION}"
# Les releases actuelles utilisent ComfyUI comme moteur interne principal pour
# les packs ComfyUI. Une ancienne .env sans profil ne doit pas le laisser éteint.
if [[ -z "${COMPOSE_PROFILES:-}" ]]; then
  COMPOSE_PROFILES=comfyui
  export COMPOSE_PROFILES
fi
VIDIOAI_COMFYUI_RESERVE_VRAM_GB=${VIDIOAI_COMFYUI_RESERVE_VRAM_GB:-12}
export VIDIOAI_COMFYUI_RESERVE_VRAM_GB
COMFYUI_IMAGE=${VIDIOAI_COMFYUI_IMAGE:-ghcr.io/lecode-official/comfyui-docker@sha256:e27739fc19d577d694ea99846a6c602e06dac963bebb2f056e22d97d19c392dd}

SOURCE_KIND=
SOURCE_VALUE=
MANIFEST_SOURCE=
if [[ -n "${VIDIOAI_RELEASE_ARCHIVE:-}" ]]; then
  SOURCE_KIND=local
  SOURCE_VALUE=$(expand_version "${VIDIOAI_RELEASE_ARCHIVE}")
  [[ -f "${SOURCE_VALUE}" ]] || fail "Release locale introuvable: ${SOURCE_VALUE}"
  read_release_manifest_from_archive "${SOURCE_VALUE}"
elif [[ -n "${VIDIOAI_RELEASE_URL:-}" || -n "${VIDIOAI_RELEASE_BASE_URL:-}" ]]; then
  SOURCE_KIND=url
  if [[ -n "${VIDIOAI_RELEASE_URL:-}" ]]; then
    SOURCE_VALUE=$(expand_version "${VIDIOAI_RELEASE_URL}")
  else
    SOURCE_VALUE="${VIDIOAI_RELEASE_BASE_URL%/}/${VERSION}/deployment.tar.gz"
  fi
  if [[ -n "${VIDIOAI_RELEASE_MANIFEST_URL:-}" ]]; then
    MANIFEST_SOURCE=$(expand_version "${VIDIOAI_RELEASE_MANIFEST_URL}")
  elif [[ -n "${VIDIOAI_RELEASE_BASE_URL:-}" ]]; then
    MANIFEST_SOURCE="${VIDIOAI_RELEASE_BASE_URL%/}/${VERSION}/release.json"
  else
    MANIFEST_SOURCE="${SOURCE_VALUE%/*}/release.json"
  fi
  curl -fsSIL "${SOURCE_VALUE}" >/dev/null \
    || fail "Bundle distant introuvable avant installation: ${SOURCE_VALUE}"
  curl -fsSL "${MANIFEST_SOURCE}" -o "${MANIFEST_FILE}" \
    || fail "Manifest distant introuvable: ${MANIFEST_SOURCE}"
elif [[ -n "${AWS_S3_BUCKET:-}" ]]; then
  SOURCE_KIND=s3
  SOURCE_VALUE="s3://${AWS_S3_BUCKET}/releases/${VERSION}/deployment.tar.gz"
  MANIFEST_SOURCE="s3://${AWS_S3_BUCKET}/releases/${VERSION}/release.json"
  AWS_ENDPOINT_ARGS=()
  if [[ -n "${AWS_ENDPOINT_URL_S3:-}" ]]; then
    AWS_ENDPOINT_ARGS=(--endpoint-url "${AWS_ENDPOINT_URL_S3}")
  fi
  aws s3api head-object --bucket "${AWS_S3_BUCKET}" \
    --key "releases/${VERSION}/deployment.tar.gz" "${AWS_ENDPOINT_ARGS[@]}" >/dev/null \
    || fail "Bundle S3 introuvable avant installation: ${SOURCE_VALUE}"
  aws s3 cp "${MANIFEST_SOURCE}" "${MANIFEST_FILE}" "${AWS_ENDPOINT_ARGS[@]}" >/dev/null \
    || fail "Manifest S3 introuvable: ${MANIFEST_SOURCE}"
elif [[ -f "${SOURCE_PROJECT_DIR}/output/vidioai-release-${VERSION}.tar.gz" ]]; then
  SOURCE_KIND=local
  SOURCE_VALUE="${SOURCE_PROJECT_DIR}/output/vidioai-release-${VERSION}.tar.gz"
  read_release_manifest_from_archive "${SOURCE_VALUE}"
else
  fail "Aucune release ${VERSION}: configurez VIDIOAI_RELEASE_URL, AWS_S3_BUCKET ou VIDIOAI_RELEASE_ARCHIVE."
fi

# Les trois tags et leurs digests sont vérifiés avant tout arrêt, pull, build ou
# modification du répertoire installé.
validate_manifest_and_remote_images
NORMALIZED_PROFILES=",${COMPOSE_PROFILES:-},"
NORMALIZED_PROFILES=${NORMALIZED_PROFILES// /}
if [[ "${NORMALIZED_PROFILES}" == *",comfyui,"* ]]; then
  [[ "${COMFYUI_IMAGE}" == *@sha256:* ]] \
    || fail "VIDIOAI_COMFYUI_IMAGE doit être épinglée par digest quand le profil comfyui est actif."
  docker buildx imagetools inspect "${COMFYUI_IMAGE}" >/dev/null 2>&1 \
    || fail "Image ComfyUI distante introuvable avant installation: ${COMFYUI_IMAGE}"
fi

stage 2 Download
case "${SOURCE_KIND}" in
  local) cp "${SOURCE_VALUE}" "${ARCHIVE_FILE}" ;;
  url) curl -fsSL "${SOURCE_VALUE}" -o "${ARCHIVE_FILE}" ;;
  s3) aws s3 cp "${SOURCE_VALUE}" "${ARCHIVE_FILE}" "${AWS_ENDPOINT_ARGS[@]}" >/dev/null ;;
  *) fail "Source de release interne invalide: ${SOURCE_KIND}" ;;
esac

stage 3 Verify
mkdir -p "${EXTRACT_DIR}"
while IFS= read -r entry; do
  [[ "${entry}" != /* && "${entry}" != *"../"* && "${entry}" != *"/.."* ]] \
    || fail "Archive dangereuse refusée: ${entry}"
  [[ "${entry}" == "release-${VERSION}" || "${entry}" == "release-${VERSION}/"* ]] \
    || fail "Entrée hors release-${VERSION} refusée: ${entry}"
done < <(tar -tzf "${ARCHIVE_FILE}")
tar -xzf "${ARCHIVE_FILE}" -C "${EXTRACT_DIR}"
RELEASE_DIR="${EXTRACT_DIR}/release-${VERSION}"
[[ -f "${RELEASE_DIR}/release.json" && -f "${RELEASE_DIR}/SHA256SUMS" ]] \
  || fail "Bundle incomplet: manifest ou SHA256SUMS absent."
cmp -s "${MANIFEST_FILE}" "${RELEASE_DIR}/release.json" \
  || fail "Le manifest téléchargé ne correspond pas au bundle."
(cd "${RELEASE_DIR}" && sha256sum --check SHA256SUMS)
# `sha256sum --check` ignore les fichiers non listés. Refuser explicitement tout
# ajout à l'archive empêche donc qu'un script d'installation non signé soit
# injecté à côté des fichiers attendus.
(
  cd "${RELEASE_DIR}"
  find . -type f ! -path './SHA256SUMS' -print \
    | sed 's#^\./##' \
    | LC_ALL=C sort > "${WORK_DIR}/bundle-files.actual"
  awk 'length($0) >= 67 { print substr($0, 67) }' SHA256SUMS \
    | LC_ALL=C sort > "${WORK_DIR}/bundle-files.signed"
)
cmp -s "${WORK_DIR}/bundle-files.actual" "${WORK_DIR}/bundle-files.signed" \
  || fail "Bundle invalide: la liste des fichiers ne correspond pas à SHA256SUMS."
validate_manifest_and_remote_images

stage 4 Build
echo "Release immuable: aucun build local; téléchargement des trois images vérifiées."
REGISTRY=$(jq -r '.registry' "${MANIFEST_FILE}")
for service in backend frontend worker; do
  docker pull "${REGISTRY}/${service}:${VERSION}"
done
if [[ "${NORMALIZED_PROFILES}" == *",comfyui,"* ]]; then
  docker pull "${COMFYUI_IMAGE}"
fi

stage 5 Install
if [[ -f "${INSTALL_DIR}/.current-version" && -x "${INSTALL_DIR}/deploy/scripts/shutdown.sh" ]]; then
  VIDIOAI_PROJECT_DIR="${INSTALL_DIR}" \
  VIDIOAI_ENV_FILE="${ENV_FILE}" \
  VIDIOAI_STOP_HOST_AGENT=false \
    "${INSTALL_DIR}/deploy/scripts/shutdown.sh"
fi
mkdir -p "${INSTALL_DIR}"
# Les permissions exécutables sont conservées sans répliquer sur l'hôte les UID
# numériques de la machine CI ayant fabriqué l'archive.
cp -R "${RELEASE_DIR}/." "${INSTALL_DIR}/"
if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${INSTALL_DIR}/.env.production.example" "${ENV_FILE}"
fi
upsert_env VIDIOAI_VERSION "${VERSION}"
upsert_env VIDIOAI_REGISTRY "${REGISTRY}"
upsert_env COMPOSE_PROFILES "${COMPOSE_PROFILES}"
upsert_env VIDIOAI_COMFYUI_RESERVE_VRAM_GB "${VIDIOAI_COMFYUI_RESERVE_VRAM_GB}"

if [[ "${VIDIOAI_RUN_BOOTSTRAP:-false}" == "true" ]]; then
  "${INSTALL_DIR}/deploy/scripts/bootstrap-server.sh"
fi

stage 6 Start
VIDIOAI_PROJECT_DIR="${INSTALL_DIR}" \
VIDIOAI_ENV_FILE="${ENV_FILE}" \
VIDIOAI_COMPOSE_FILE="${COMPOSE_FILE}" \
VIDIOAI_SKIP_SMOKE_TEST=true \
  "${INSTALL_DIR}/deploy/scripts/deploy.sh" "${VERSION}"

stage 7 Healthcheck
BASE_URL=${VIDIOAI_RELEASE_HEALTH_URL:-http://127.0.0.1:${VIDIOAI_HTTP_PORT:-8080}}
BACKEND_HEALTH=$(curl -fsS "${BASE_URL}/api/health")
jq -e --arg version "${VERSION}" '.version == $version' <<<"${BACKEND_HEALTH}" >/dev/null \
  || fail "Le backend sain n'expose pas la version demandée ${VERSION}."

# Le healthcheck Worker ne déclenche ni import CUDA ni génération. Le contrôle
# s'effectue dans le conteneur, car le service interne n'est volontairement pas
# publié sur l'hôte de production.
WORKER_HEALTH=$(VIDIOAI_VERSION="${VERSION}" docker compose \
  -f "${COMPOSE_FILE}" \
  --env-file "${ENV_FILE}" \
  exec -T worker \
  curl -fsS http://127.0.0.1:8000/health)
jq -e --arg version "${VERSION}" '.version == $version' <<<"${WORKER_HEALTH}" >/dev/null \
  || fail "Le worker sain n'expose pas la version demandée ${VERSION}."
curl -fsS "${BASE_URL}/" >/dev/null

DEPLOY_SUCCEEDED=true
printf 'VidioAI %s deployed successfully\n' "${VERSION}"
