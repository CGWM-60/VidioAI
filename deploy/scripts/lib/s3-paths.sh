#!/usr/bin/env bash

# Valide exclusivement le nom du bucket. Un préfixe (`bucket/releases`) ou une
# URI complète provoquerait sinon silencieusement `bucket/bucket/releases`.
vidioai_validate_s3_bucket() {
  local bucket=${1:-}
  if [[ ! "${bucket}" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ || "${bucket}" == *"/"* || "${bucket}" == s3://* ]]; then
    echo "AWS_S3_BUCKET doit contenir uniquement le nom du bucket, sans s3:// ni préfixe: ${bucket}" >&2
    return 1
  fi
}

vidioai_release_uri() {
  local bucket=${1:?bucket requis}
  local version=${2:?version requise}
  local filename=${3:?fichier requis}
  vidioai_validate_s3_bucket "${bucket}" || return 1
  printf 's3://%s/releases/%s/%s' "${bucket}" "${version}" "${filename}"
}
