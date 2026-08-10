#!/usr/bin/env bash
set -Eeuo pipefail

# Mesure reproductible de coût/latence. Le tarif horaire est volontairement un
# paramètre : les prix cloud évoluent et ne doivent pas être figés dans le code.
BASE_URL=${1:-http://127.0.0.1}
: "${HOURLY_PRICE_EUR:?Définissez HOURLY_PRICE_EUR pour la machine testée}"
RUNS=${BENCHMARK_RUNS:-5}
MODE=${BENCHMARK_MODE:-TEXT_TO_VIDEO}
MODEL=${BENCHMARK_MODEL:-vidio-motion-local}

total_ms=0
for ((run=1; run<=RUNS; run++)); do
  started=$(date +%s%3N)
  generation=$(curl -fsS -X POST "${BASE_URL}/api/videos/generate" \
    -H 'content-type: application/json' \
    -d "{\"mode\":\"${MODE}\",\"prompt\":\"Benchmark VidioAI travelling cinématographique\",\"model_id\":\"${MODEL}\",\"duration_seconds\":4,\"resolution\":\"720p\",\"audio\":false}")
  id=$(jq -r '.id' <<<"${generation}")
  while true; do
    status=$(curl -fsS "${BASE_URL}/api/generations/${id}" | jq -r '.status')
    [[ "${status}" == "COMPLETED" ]] && break
    [[ "${status}" == "FAILED" || "${status}" == "CANCELLED" ]] && { echo "Job ${id}: ${status}" >&2; exit 1; }
    sleep 1
  done
  elapsed=$(( $(date +%s%3N) - started ))
  total_ms=$((total_ms + elapsed))
  echo "run=${run} latency_ms=${elapsed}"
done

average_ms=$((total_ms / RUNS))
cost=$(awk -v ms="${average_ms}" -v hourly="${HOURLY_PRICE_EUR}" 'BEGIN { printf "%.6f", (ms / 3600000) * hourly }')
echo "average_latency_ms=${average_ms} estimated_compute_cost_eur=${cost} runs=${RUNS}"
