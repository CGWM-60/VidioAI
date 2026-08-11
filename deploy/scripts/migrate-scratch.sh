#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${EUID}" -eq 0 ]] || { echo "Exécutez la migration avec sudo." >&2; exit 1; }
PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
OLD_SCRATCH=${VIDIOAI_OLD_SCRATCH_DIR:-/var/lib/vidioai/scratch}
ACTION=${1:-prepare}
source "${PROJECT_DIR}/deploy/scripts/lib/scratch-storage.sh"

vidioai_require_production_scratch "${ENV_FILE}"
vidioai_assert_distinct_scratch_filesystem "${VIDIOAI_SCRATCH_DIR}"
[[ "${OLD_SCRATCH}" == "/var/lib/vidioai/scratch" ]] || {
  echo "Ancien chemin inattendu: ${OLD_SCRATCH}" >&2
  exit 1
}
marker="${VIDIOAI_SCRATCH_DIR}/.vidioai-migration-verified"
runtime_marker="${VIDIOAI_SCRATCH_DIR}/.vidioai-migration-runtime-verified"

compare_trees() {
  local differences
  differences=$(rsync -aHAXnci "${OLD_SCRATCH}/" "${VIDIOAI_SCRATCH_DIR}/" \
    --exclude '.vidioai-migration-verified' \
    --exclude '.vidioai-migration-runtime-verified')
  [[ -z "${differences}" ]] || {
    echo "La comparaison checksum ancien/nouveau Scratch a détecté des différences." >&2
    return 1
  }
}

case "${ACTION}" in
  prepare)
    [[ -d "${OLD_SCRATCH}" ]] || { echo "Aucun ancien Scratch à migrer."; exit 0; }
    if docker ps --filter label=com.docker.compose.service=worker -q | grep -q .; then
      echo "Un Worker tourne encore; arrêtez-le avant la copie Scratch." >&2
      exit 1
    fi
    install -d -m 0770 -o root -g docker \
      "${VIDIOAI_SCRATCH_DIR}"/{models,cache,work,worker-work}
    rsync -aHAX --numeric-ids "${OLD_SCRATCH}/" "${VIDIOAI_SCRATCH_DIR}/"
    compare_trees
    setfacl -Rm u:10001:rwx,u:10002:rwx "${VIDIOAI_SCRATCH_DIR}"
    setfacl -Rdm u:10001:rwx,u:10002:rwx "${VIDIOAI_SCRATCH_DIR}"
    old_bytes=$(find "${OLD_SCRATCH}" -type f -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
    new_bytes=$(find "${VIDIOAI_SCRATCH_DIR}" -type f ! -name '.vidioai-migration-*' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')
    old_files=$(find "${OLD_SCRATCH}" -type f | wc -l)
    new_files=$(find "${VIDIOAI_SCRATCH_DIR}" -type f ! -name '.vidioai-migration-*' | wc -l)
    [[ "${new_bytes}" -ge "${old_bytes}" && "${new_files}" -ge "${old_files}" ]] || {
      echo "Comptage migration incohérent: ancien=${old_files}/${old_bytes}, nouveau=${new_files}/${new_bytes}." >&2
      exit 1
    }
    printf 'old_bytes=%s\nnew_bytes=%s\nold_files=%s\nnew_files=%s\n' \
      "${old_bytes}" "${new_bytes}" "${old_files}" "${new_files}" >"${marker}"
    echo "MIGRATION_PREPARED ancien=${OLD_SCRATCH} nouveau=${VIDIOAI_SCRATCH_DIR}; ancien conservé."
    ;;
  verify)
    [[ -f "${marker}" ]] || { echo "Migration prepare non validée." >&2; exit 1; }
    compare_trees
    "${PROJECT_DIR}/deploy/scripts/verify-scratch.sh" worker >/dev/null
    touch "${runtime_marker}"
    echo "MIGRATION_RUNTIME_VERIFIED; le nettoyage explicite peut être autorisé."
    ;;
  cleanup)
    [[ -f "${runtime_marker}" ]] || { echo "Validation runtime absente; nettoyage refusé." >&2; exit 1; }
    [[ "${VIDIOAI_CONFIRM_SCRATCH_CLEANUP:-}" == "DELETE_OLD_SCRATCH" ]] || {
      echo "Définissez VIDIOAI_CONFIRM_SCRATCH_CLEANUP=DELETE_OLD_SCRATCH pour confirmer." >&2
      exit 1
    }
    compare_trees
    rm -rf --one-file-system -- "${OLD_SCRATCH}"
    echo "Ancien Scratch supprimé après double validation: ${OLD_SCRATCH}"
    ;;
  *) echo "Usage: migrate-scratch.sh prepare|verify|cleanup" >&2; exit 2 ;;
esac
