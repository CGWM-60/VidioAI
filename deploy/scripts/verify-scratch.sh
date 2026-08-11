#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${VIDIOAI_PROJECT_DIR:-/opt/vidioai}
ENV_FILE=${VIDIOAI_ENV_FILE:-${PROJECT_DIR}/.env.production}
COMPOSE_FILE=${VIDIOAI_COMPOSE_FILE:-${PROJECT_DIR}/compose.production.yml}
MODE=${1:-host}
source "${PROJECT_DIR}/deploy/scripts/lib/scratch-storage.sh"

vidioai_require_production_scratch "${ENV_FILE}"
vidioai_assert_distinct_scratch_filesystem "${VIDIOAI_SCRATCH_DIR}"

read -r scratch_filesystem scratch_total_bytes scratch_available_bytes scratch_target < <(
  df -B1 --output=source,size,avail,target "${VIDIOAI_SCRATCH_DIR}" | tail -n 1
)
scratch_device=$(stat -c '%d' "${VIDIOAI_SCRATCH_DIR}")
minimum_total_bytes=${VIDIOAI_MIN_SCRATCH_TOTAL_BYTES:-214748364800}
[[ "${scratch_total_bytes}" -ge "${minimum_total_bytes}" ]] || {
  echo "Scratch trop petit: ${scratch_total_bytes} octets, minimum ${minimum_total_bytes}." >&2
  exit 1
}

worker_mount_ok=true
worker_filesystem=""
worker_total_bytes=0
worker_available_bytes=0
if [[ "${MODE}" == "worker" ]]; then
  worker_id=$(VIDIOAI_SCRATCH_DIR="${VIDIOAI_SCRATCH_DIR}" \
    docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" ps -q worker)
  [[ -n "${worker_id}" ]] || { echo "Conteneur Worker absent." >&2; exit 1; }

  mounts=$(docker inspect "${worker_id}" --format '{{json .Mounts}}')
  jq -e --arg scratch "${VIDIOAI_SCRATCH_DIR}" '
    any(.[]; .Source == ($scratch + "/models") and .Destination == "/models") and
    any(.[]; .Source == ($scratch + "/cache") and .Destination == "/cache") and
    any(.[]; .Source == ($scratch + "/work") and .Destination == "/work") and
    any(.[]; .Source == ($scratch + "/worker-work") and .Destination == "/worker-work")
  ' <<<"${mounts}" >/dev/null || worker_mount_ok=false

  worker_probe=$(docker exec "${worker_id}" python -c '
import json, os, pathlib
paths = ["/models", "/cache", "/work", "/worker-work"]
stats = {path: os.stat(path).st_dev for path in paths}
root_device = os.stat("/").st_dev
write_ok = True
for raw in paths:
    probe = pathlib.Path(raw) / ".vidioai-scratch-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        write_ok = False
vfs = os.statvfs("/models")
print(json.dumps({
    "devices": stats,
    "root_device": root_device,
    "same_device": len(set(stats.values())) == 1,
    "distinct_from_root": stats["/models"] != root_device,
    "write_ok": write_ok,
    "total_bytes": vfs.f_frsize * vfs.f_blocks,
    "available_bytes": vfs.f_frsize * vfs.f_bavail,
}))
')
  jq -e '.same_device == true and .distinct_from_root == true and .write_ok == true' \
    <<<"${worker_probe}" >/dev/null || worker_mount_ok=false
  worker_total_bytes=$(jq -r '.total_bytes' <<<"${worker_probe}")
  worker_available_bytes=$(jq -r '.available_bytes' <<<"${worker_probe}")
  worker_filesystem=$(jq -r '.devices["/models"]' <<<"${worker_probe}")
  [[ "${worker_total_bytes}" -eq "${scratch_total_bytes}" ]] || worker_mount_ok=false
fi

scratch_mount_ok=true
[[ "${worker_mount_ok}" == "true" ]] || scratch_mount_ok=false
jq -n \
  --argjson scratch_mount_ok "${scratch_mount_ok}" \
  --arg scratch_filesystem "${scratch_filesystem}" \
  --arg scratch_target "${scratch_target}" \
  --arg scratch_device "${scratch_device}" \
  --argjson scratch_total_bytes "${scratch_total_bytes}" \
  --argjson scratch_available_bytes "${scratch_available_bytes}" \
  --argjson scratch_minimum_bytes "${minimum_total_bytes}" \
  --argjson worker_mount_ok "${worker_mount_ok}" \
  --arg worker_filesystem "${worker_filesystem}" \
  --argjson worker_total_bytes "${worker_total_bytes}" \
  --argjson worker_available_bytes "${worker_available_bytes}" \
  '{scratch_mount_ok:$scratch_mount_ok,scratch_filesystem:$scratch_filesystem,
    scratch_target:$scratch_target,scratch_device:$scratch_device,
    scratch_total_bytes:$scratch_total_bytes,scratch_available_bytes:$scratch_available_bytes,
    scratch_minimum_bytes:$scratch_minimum_bytes,
    worker_mount_ok:$worker_mount_ok,worker_filesystem:$worker_filesystem,
    worker_total_bytes:$worker_total_bytes,worker_available_bytes:$worker_available_bytes}'
[[ "${scratch_mount_ok}" == "true" ]] || exit 1
