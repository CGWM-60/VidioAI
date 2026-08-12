#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-.}
cd "$ROOT"

grep -q '^VIDIOAI_VERSION=2026.08.11-12$' deploy/config/production.env.example
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

# Pas de support artificiel par nom de modèle/repository dans le runtime.
if grep -R -n -E 'MiniMaxH3|MiniMax-H3|MiniMax/H3' \
  worker/app backend/src 2>/dev/null; then
  echo "ERREUR: branchement H3 codé en dur détecté."
  exit 1
fi

echo "Contrat ModularPipeline VidioAI 2026.08.11-12: OK"
