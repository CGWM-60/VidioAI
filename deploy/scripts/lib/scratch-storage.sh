#!/usr/bin/env bash

# Contrat unique du stockage lourd GPU. Ce fichier est sourcé par bootstrap,
# preflight, deploy, rollback et les diagnostics d'acceptance.

vidioai_env_value() {
  local env_file=${1:?fichier env requis}
  local key=${2:?clé requise}
  sed -n "s/^${key}=//p" "${env_file}" | tail -n 1
}

vidioai_require_production_scratch() {
  local env_file=${1:?fichier env requis}
  [[ -f "${env_file}" ]] || {
    echo "Configuration production absente: ${env_file}" >&2
    return 1
  }
  local configured
  configured=$(vidioai_env_value "${env_file}" "VIDIOAI_SCRATCH_DIR")
  [[ "${configured}" == "/scratch/vidioai" ]] || {
    echo "VIDIOAI_SCRATCH_DIR doit valoir explicitement /scratch/vidioai dans ${env_file}; valeur reçue: ${configured:-absente}." >&2
    return 1
  }
  if [[ -n "${VIDIOAI_SCRATCH_DIR+x}" && "${VIDIOAI_SCRATCH_DIR}" != "${configured}" ]]; then
    echo "VIDIOAI_SCRATCH_DIR du shell (${VIDIOAI_SCRATCH_DIR}) contredit ${env_file} (${configured})." >&2
    return 1
  fi
  export VIDIOAI_SCRATCH_DIR="${configured}"
}

vidioai_assert_distinct_scratch_filesystem() {
  local scratch_dir=${1:?répertoire Scratch requis}
  [[ "${scratch_dir}" == "/scratch/vidioai" ]] || {
    echo "Scratch production invalide: ${scratch_dir}" >&2
    return 1
  }
  [[ -d "${scratch_dir}" && -w "${scratch_dir}" ]] || {
    echo "Scratch absent ou non inscriptible: ${scratch_dir}" >&2
    return 1
  }
  command -v findmnt >/dev/null 2>&1 || {
    echo "findmnt est requis pour vérifier le filesystem Scratch." >&2
    return 1
  }
  local scratch_target root_device scratch_device
  scratch_target=$(findmnt -T "${scratch_dir}" -n -o TARGET)
  root_device=$(stat -c '%d' /)
  scratch_device=$(stat -c '%d' "${scratch_dir}")
  [[ "${scratch_target}" == "/scratch" || "${scratch_target}" == /scratch/* ]] || {
    echo "${scratch_dir} résout vers le mount ${scratch_target:-inconnu}, pas vers /scratch." >&2
    return 1
  }
  [[ "${scratch_device}" != "${root_device}" ]] || {
    echo "${scratch_dir} utilise le même filesystem que /; stockage lourd refusé." >&2
    return 1
  }
}

