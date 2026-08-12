#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-.}
cd "$ROOT"

test -f worker/app/model_bundle.py
test -f worker/app/inference_recipe.py
test -f worker/tests/test_release_2026_08_11_12.py

grep -q '^VIDIOAI_VERSION=2026.08.11-12$' deploy/config/production.env.example
grep -q '^VIDIOAI_AUTO_CACHE_MODELS=false$' deploy/config/production.env.example
grep -q 'VIDIOAI_AUTO_CACHE_MODELS: ${VIDIOAI_AUTO_CACHE_MODELS:-false}' compose.production.yml

grep -q 'auto_cache_models_enabled()' backend/src/platform.rs
grep -q 'state.object_storage.enabled() && auto_cache_models_enabled()' backend/src/platform.rs
grep -q '"CACHE_MANUAL"' backend/src/platform.rs

grep -q 'pub bundle: serde_json::Value' backend/src/platform.rs
grep -q 'pub bundle: Option<serde_json::Value>' backend/src/worker.rs

grep -q 'peft==0.19.1' worker/requirements.txt
grep -q 'ModelBundleManager' worker/app/runtime.py
grep -q 'InferenceRecipeResolver' worker/app/runtime.py
grep -q '"true_cfg_scale"' worker/app/adapters/image_to_image.py
grep -q '"true_cfg_scale"' worker/app/adapters/image_to_video.py

grep -q 'quality: imageQuality' frontend/app/images/page.js
grep -q 'Appliquer LoRA / recette' 'frontend/app/models/[id]/page.js'

echo "Contrat release 2026.08.11-12: OK"
