#!/usr/bin/env bash
set -Eeuo pipefail

# Bootstrap idempotent d'un hôte Ubuntu/Debian. Chaque étape vérifie son état
# avant mutation pour permettre une relance après une interruption réseau.
if [[ "${EUID}" -ne 0 ]]; then
  echo "Exécutez ce script avec sudo." >&2
  exit 1
fi

source /etc/os-release
case "${ID}" in
  ubuntu|debian) ;;
  *) echo "Distribution non supportée: ${ID}" >&2; exit 1 ;;
esac

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y acl ca-certificates curl ffmpeg file gnupg jq git openssl pciutils rsync unzip util-linux

# Ubuntu Noble ne publie plus nécessairement le paquet `awscli`. L'installeur
# officiel v2 est autonome et fonctionne aussi bien en x86_64 qu'en aarch64.
if ! command -v aws >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64|amd64) AWS_CLI_ARCH=x86_64 ;;
    aarch64|arm64) AWS_CLI_ARCH=aarch64 ;;
    *) echo "Architecture AWS CLI non supportée: $(uname -m)" >&2; exit 1 ;;
  esac
  (
    AWS_CLI_TMP=$(mktemp -d /tmp/vidioai-awscli.XXXXXX)
    # La cible est un répertoire temporaire créé ci-dessus, jamais un chemin
    # fourni par l'environnement ou un répertoire système large.
    trap 'rm -rf -- "${AWS_CLI_TMP}"' EXIT
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_CLI_ARCH}.zip" \
      -o "${AWS_CLI_TMP}/awscliv2.zip"
    unzip -q "${AWS_CLI_TMP}/awscliv2.zip" -d "${AWS_CLI_TMP}"
    "${AWS_CLI_TMP}/aws/install" --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli
  )
fi
aws --version >/dev/null

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

# Le runtime NVIDIA n'est installé que si une carte NVIDIA est détectée. Un
# serveur CPU reste donc parfaitement valide pour le moteur FFmpeg local.
if lspci 2>/dev/null | grep -qi nvidia && ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
  curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

PRODUCTION_ENV=${VIDIOAI_ENV_FILE:-/opt/vidioai/.env.production}
SCRATCH_DIR=/scratch/vidioai
OLD_SCRATCH_DIR=/var/lib/vidioai/scratch

# Une simple existence de /scratch ne suffit pas : il doit s'agir d'un mount
# distinct du filesystem racine, sinon les poids retomberaient sur /dev/sda1.
[[ -d /scratch && -w /scratch ]] || {
  echo "/scratch est absent ou non inscriptible; bootstrap GPU refusé." >&2
  exit 1
}
SCRATCH_TARGET=$(findmnt -T /scratch -n -o TARGET)
ROOT_DEVICE=$(stat -c '%d' /)
SCRATCH_DEVICE=$(stat -c '%d' /scratch)
SCRATCH_TOTAL_BYTES=$(df -B1 --output=size /scratch | tail -n 1 | tr -d '[:space:]')
MIN_SCRATCH_TOTAL_BYTES=${VIDIOAI_MIN_SCRATCH_TOTAL_BYTES:-214748364800}
[[ "${SCRATCH_TARGET}" == "/scratch" || "${SCRATCH_TARGET}" == /scratch/* ]] || {
  echo "/scratch résout vers ${SCRATCH_TARGET:-inconnu}; mount Scratch distinct requis." >&2
  exit 1
}
[[ "${ROOT_DEVICE}" != "${SCRATCH_DEVICE}" ]] || {
  echo "/scratch utilise le filesystem racine; stockage modèles refusé." >&2
  exit 1
}
[[ "${SCRATCH_TOTAL_BYTES}" -ge "${MIN_SCRATCH_TOTAL_BYTES}" ]] || {
  echo "/scratch est trop petit (${SCRATCH_TOTAL_BYTES} octets); minimum ${MIN_SCRATCH_TOTAL_BYTES}." >&2
  exit 1
}

install -d -m 0750 -o root -g docker /opt/vidioai /var/lib/vidioai/{state,outputs,backups}
install -d -m 0770 -o root -g docker "${SCRATCH_DIR}"/{models,cache,work,worker-work}
# UID 10001 (backend) et 10002 (worker) partagent le Scratch sans exécuter les
# conteneurs en root. Les ACL par défaut s'appliquent aussi aux nouveaux poids.
setfacl -Rm u:10001:rwx,u:10002:rwx /var/lib/vidioai/{state,outputs} "${SCRATCH_DIR}"
setfacl -Rdm u:10001:rwx,u:10002:rwx /var/lib/vidioai/{state,outputs} "${SCRATCH_DIR}"

# Migration automatique sûre : copie et comparaison checksum, jamais de
# suppression de l'ancien stockage pendant le bootstrap.
if [[ -d "${OLD_SCRATCH_DIR}" ]] \
    && find "${OLD_SCRATCH_DIR}" -mindepth 1 -print -quit | grep -q .; then
  if docker ps --filter label=com.docker.compose.service=worker -q | grep -q .; then
    echo "Un Worker tourne encore; migration automatique Scratch refusée." >&2
    exit 1
  fi
  rsync -aHAX --numeric-ids "${OLD_SCRATCH_DIR}/" "${SCRATCH_DIR}/"
  MIGRATION_DIFF=$(rsync -aHAXnci "${OLD_SCRATCH_DIR}/" "${SCRATCH_DIR}/" \
    --exclude '.vidioai-migration-*')
  [[ -z "${MIGRATION_DIFF}" ]] || {
    echo "Migration Scratch copiée mais comparaison checksum en échec; ancien stockage conservé." >&2
    exit 1
  }
  printf 'source=%s\ntarget=%s\nverified_at=%s\n' \
    "${OLD_SCRATCH_DIR}" "${SCRATCH_DIR}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"${SCRATCH_DIR}/.vidioai-migration-verified"
  echo "Ancien Scratch copié et vérifié; ${OLD_SCRATCH_DIR} est conservé."
fi
setfacl -Rm u:10001:rwx,u:10002:rwx "${SCRATCH_DIR}"
setfacl -Rdm u:10001:rwx,u:10002:rwx "${SCRATCH_DIR}"

install -d -m 0750 -o root -g root "$(dirname "${PRODUCTION_ENV}")"
touch "${PRODUCTION_ENV}"
if grep -q '^VIDIOAI_SCRATCH_DIR=' "${PRODUCTION_ENV}"; then
  sed -i 's#^VIDIOAI_SCRATCH_DIR=.*#VIDIOAI_SCRATCH_DIR=/scratch/vidioai#' "${PRODUCTION_ENV}"
else
  printf 'VIDIOAI_SCRATCH_DIR=/scratch/vidioai\n' >>"${PRODUCTION_ENV}"
fi

# Le binaire natif est livré dans l'archive de release. Il n'est jamais copié
# dans une image Docker, ce qui garantit que sysinfo et nvidia-smi voient l'hôte.
HOST_AGENT_BINARY=${HOST_AGENT_BINARY:-/opt/vidioai/deploy/bin/vidioai-host-agent}
HOST_AGENT_SERVICE=${HOST_AGENT_SERVICE:-/opt/vidioai/deploy/systemd/vidioai-host-agent.service}
HOST_AGENT_ENV=/etc/vidioai/host-agent.env
test -x "${HOST_AGENT_BINARY}" || { echo "Binaire Host Agent absent: ${HOST_AGENT_BINARY}" >&2; exit 1; }
test -f "${HOST_AGENT_SERVICE}" || { echo "Service systemd absent: ${HOST_AGENT_SERVICE}" >&2; exit 1; }

if ! id vidioai-host-agent >/dev/null 2>&1; then
  useradd --system --home-dir /nonexistent --shell /usr/sbin/nologin vidioai-host-agent
fi
install -m 0755 "${HOST_AGENT_BINARY}" /usr/local/bin/vidioai-host-agent
install -m 0644 "${HOST_AGENT_SERVICE}" /etc/systemd/system/vidioai-host-agent.service
install -d -m 0750 -o root -g vidioai-host-agent /etc/vidioai

# Un même secret protège l'agent et est injecté dans le backend via Compose.
# Lorsqu'il manque, le bootstrap le génère et complète le fichier production.
HOST_AGENT_TOKEN=${HOST_AGENT_TOKEN:-}
if [[ -z "${HOST_AGENT_TOKEN}" && -f "${PRODUCTION_ENV}" ]]; then
  HOST_AGENT_TOKEN=$(sed -n 's/^HOST_AGENT_TOKEN=//p' "${PRODUCTION_ENV}" | tail -n 1)
fi
if [[ ! "${HOST_AGENT_TOKEN}" =~ ^[A-Za-z0-9._-]{32,}$ || "${HOST_AGENT_TOKEN}" == replace-with-* ]]; then
  HOST_AGENT_TOKEN=$(openssl rand -hex 32)
fi

# Synchroniser systématiquement le secret choisi, y compris lorsqu'il a été
# fourni au script. Auparavant ce cas pouvait laisser Compose et systemd avec
# deux tokens différents et provoquer des 401 intermittents.
if grep -q '^HOST_AGENT_TOKEN=' "${PRODUCTION_ENV}"; then
  sed -i "s/^HOST_AGENT_TOKEN=.*/HOST_AGENT_TOKEN=${HOST_AGENT_TOKEN}/" "${PRODUCTION_ENV}"
else
  if ! grep -q '^HOST_AGENT_URL=' "${PRODUCTION_ENV}"; then
    printf '\nHOST_AGENT_URL=http://host.docker.internal:8091\n' >> "${PRODUCTION_ENV}"
  fi
  printf 'HOST_AGENT_TOKEN=%s\n' "${HOST_AGENT_TOKEN}" >> "${PRODUCTION_ENV}"
fi
chmod 0640 "${PRODUCTION_ENV}"
umask 027
printf 'HOST_AGENT_BIND=0.0.0.0:8091\nHOST_AGENT_TOKEN=%s\n' \
  "${HOST_AGENT_TOKEN}" > "${HOST_AGENT_ENV}"
chown root:vidioai-host-agent "${HOST_AGENT_ENV}"
chmod 0640 "${HOST_AGENT_ENV}"
# Comparaison en mémoire uniquement : aucun secret n'est écrit sur stdout.
PRODUCTION_HOST_TOKEN=$(sed -n 's/^HOST_AGENT_TOKEN=//p' "${PRODUCTION_ENV}" | tail -n 1)
SERVICE_HOST_TOKEN=$(sed -n 's/^HOST_AGENT_TOKEN=//p' "${HOST_AGENT_ENV}" | tail -n 1)
[[ "${PRODUCTION_HOST_TOKEN}" == "${SERVICE_HOST_TOKEN}" ]] || {
  echo "Le token Host Agent diffère entre Compose et systemd." >&2
  exit 1
}
systemctl daemon-reload
systemctl enable --now vidioai-host-agent.service

for attempt in {1..30}; do
  if curl -fsS -H "X-VidioAI-Host-Token: ${HOST_AGENT_TOKEN}" \
    http://127.0.0.1:8091/health >/dev/null; then
    break
  fi
  [[ ${attempt} -eq 30 ]] && { journalctl -u vidioai-host-agent --no-pager -n 80 >&2; exit 1; }
  sleep 1
done
curl -fsS -H "X-VidioAI-Host-Token: ${HOST_AGENT_TOKEN}" \
  http://127.0.0.1:8091/system | jq -e '.source == "host" and .system and .cpu and .ram and .storage' >/dev/null
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA absente : ce serveur ne peut pas utiliser GPU_PRODUCTION." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker run --rm --gpus all "${CUDA_TEST_IMAGE:-nvidia/cuda:12.8.1-base-ubuntu24.04}" nvidia-smi >/dev/null
df -h "${SCRATCH_DIR}"
echo "Bootstrap terminé. Docker: $(docker --version) · AWS CLI: $(aws --version 2>&1)"
