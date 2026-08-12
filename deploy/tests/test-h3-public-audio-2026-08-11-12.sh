#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${1:-.}
cd "$ROOT"

grep -q '^VIDIOAI_VERSION=2026.08.11-12$' deploy/config/production.env.example
grep -q '3a2f35d4efa4c059c8bfb3bc0d6c906264895c81' worker/requirements.txt
grep -q '^torchao==0.17.0$' worker/requirements.txt
grep -q '^av==18.0.0$' worker/requirements.txt

grep -q 'struct HardwareEstimate' backend/src/hardware_estimator.rs
! grep -A12 'Repository ModularPipeline détecté' backend/src/hardware_estimator.rs \
  | grep -q 'estimate.source ='

grep -q 'MiniMaxH3Adapter' worker/app/adapters/registry.py
grep -q 'H3_ARCHITECTURE = "MiniMaxH3ModularPipeline"' worker/app/adapters/minimax_h3.py
grep -q 'workflow_for_capability' worker/app/adapters/minimax_h3.py
grep -q 'NATIVE_AUDIO_MISSING' worker/app/audio_output.py
grep -q 'OUTPUT_AUDIO_STREAM_MISSING' worker/app/audio_output.py
grep -q 'actual_audio' backend/src/worker.rs
grep -q 'actual_audio' backend/src/platform.rs
grep -q '"audio": audio' backend/src/worker.rs
grep -q 'Audio natif du modèle' frontend/app/generations/page.js

# Un vrai snapshot modular doit être accepté sans model_index classique.
grep -q 'modular_model_index.json' worker/app/runtime.py
grep -q 'MODEL_MANIFEST_MISSING' worker/app/runtime.py

# Pas de branchement par repository H3 dans worker/app.
if grep -R -n 'MiniMaxAI/MiniMax-H3' worker/app backend/src 2>/dev/null; then
  echo "ERREUR: repo-id MiniMax H3 codé en dur dans le runtime."
  exit 1
fi

echo "Contrat H3 public + audio VidioAI 2026.08.11-12: OK"
