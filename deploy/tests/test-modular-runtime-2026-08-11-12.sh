#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-.}
cd "$ROOT"

# Ce test historique verrouille ModularPipeline, pas la release active.
grep -Eq '^VIDIOAI_VERSION=[A-Za-z0-9][A-Za-z0-9._-]+$' deploy/config/production.env.example
grep -q '^VIDIOAI_AUTO_CACHE_MODELS=false$' deploy/config/production.env.example

grep -q 'modular_model_index.json' backend/src/huggingface_catalog.rs
grep -q 'Diffusers ModularPipeline' backend/src/huggingface_catalog.rs
grep -q 'pub discovered: bool' backend/src/platform.rs
grep -q 'pub downloadable: bool' backend/src/platform.rs
grep -q 'pub hardware_compatibility: String' backend/src/platform.rs
grep -q 'is_modular: bool' backend/src/worker.rs
grep -q 'modular_model_index.json' backend/src/platform.rs

grep -q 'ModularManifestResolver' worker/app/runtime.py
grep -q 'is_modular: bool = False' worker/app/schemas.py
grep -q 'ModularDiffusersAdapter' worker/app/adapters/registry.py
grep -q 'ComponentsManager' worker/app/adapters/modular_diffusers.py
grep -q 'ModularPipeline' worker/app/pipeline_resolver.py
grep -q 'modular_model_index.json' worker/app/adapters/inspectors.py
grep -q 'blocks' worker/app/capability_resolver.py

# Pas de support artificiel par nom de modèle/repository dans le runtime
# central. Les plugins/adapters de famille restent autorisés.
if grep -n -E 'MiniMaxAI/MiniMax-H3|MiniMax/H3' \
  worker/app/runtime.py worker/app/pipeline_resolver.py backend/src/*.rs 2>/dev/null; then
  echo "ERREUR: repo-id H3 codé en dur détecté."
  exit 1
fi

echo "Contrat ModularPipeline VidioAI 2026.08.11-12: OK"
