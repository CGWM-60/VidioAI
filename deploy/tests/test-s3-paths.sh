#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${PROJECT_DIR}/deploy/scripts/lib/s3-paths.sh"

EXPECTED='s3://vidioai-production/releases/2026.08.10-5/deployment.tar.gz'
ACTUAL=$(vidioai_release_uri vidioai-production 2026.08.10-5 deployment.tar.gz)
[[ "${ACTUAL}" == "${EXPECTED}" ]]

# Régression du défaut observé : la variable ne doit jamais accepter un bucket
# déjà préfixé par lui-même, par `releases` ou par le schéma S3.
for invalid in \
  'vidioai-production/vidioai-production' \
  'vidioai-production/releases' \
  's3://vidioai-production'; do
  if vidioai_validate_s3_bucket "${invalid}" 2>/dev/null; then
    echo "Bucket invalide accepté: ${invalid}" >&2
    exit 1
  fi
done

echo "Chemins S3 de release validés."
