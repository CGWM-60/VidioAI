export const INSTALL_PHASES = [
  ["hugging_face", "Téléchargement HF", "Snapshot reçu depuis Hugging Face ou restauré depuis le cache"],
  ["installation", "Installation", "Composants et dépendances installés localement"],
  ["validation", "Validation", "ModelPack, workflow et fichiers vérifiés"],
  ["cloud_backup", "Sauvegarde S3", "Étape cloud indépendante de l’installation locale"],
];

// Alias conservé pendant les rolling deploys et pour les imports existants.
export const INSTALL_STEPS = INSTALL_PHASES;

const CLOUD_BACKUP_ALIASES = {
  CACHE_PENDING: "PENDING",
  CACHE_UPLOADING: "UPLOADING",
  CACHE_READY: "COMPLETED",
  CLOUD_AVAILABLE: "COMPLETED",
  CACHE_FAILED: "FAILED",
  CACHE_CANCELLED: "CANCELLED",
  CLOUD_BACKUP_CANCELLED: "CANCELLED",
  CACHE_MANUAL: "NOT_REQUESTED",
  CACHE_DISABLED: "NOT_REQUESTED",
};

export function cloudBackupStatus(job) {
  const raw = String(job?.cloud_backup_status || job?.cache_status || "NOT_REQUESTED").toUpperCase();
  return CLOUD_BACKUP_ALIASES[raw] || raw;
}

export function cancelCloudBackupPayload(modelId) {
  return { model_id: modelId };
}

function phaseState(phase, job, localInstalled, cloudStatus) {
  const explicit = job?.phases?.[phase] || job?.phase_status?.[phase];
  if (explicit) return String(explicit).toUpperCase();

  const stage = String(job?.stage || "checking");
  const activePhase = {
    checking: "hugging_face",
    restoring_cache: "hugging_face",
    downloading: "hugging_face",
    installing: "installation",
    resolving_dependencies: "installation",
    installing_dependencies: "installation",
    validating_snapshot: "validation",
    validating_workflow: "validation",
    preflight: "validation",
    saving_cache: "cloud_backup",
    installed: "cloud_backup",
  }[stage] || "hugging_face";

  if (phase === "cloud_backup") {
    if (cloudStatus === "NOT_REQUESTED" && localInstalled) return "SKIPPED";
    return cloudStatus;
  }
  if (localInstalled) return "COMPLETED";

  const order = INSTALL_PHASES.map(([id]) => id);
  const phaseIndex = order.indexOf(phase);
  const activeIndex = order.indexOf(activePhase);
  if (job?.status === "failed" && phaseIndex === activeIndex) return "FAILED";
  if (phaseIndex < activeIndex) return "COMPLETED";
  if (phaseIndex === activeIndex) return "UPLOADING";
  return "PENDING";
}

export function installationPhases(job) {
  const cloudStatus = cloudBackupStatus(job);
  const localStatus = String(job?.local_installation_status || "").toUpperCase();
  const localInstalled = localStatus === "COMPLETED"
    || job?.local_installed === true
    || job?.stage === "installed"
    || job?.status === "completed"
    || ["UPLOADING", "COMPLETED", "FAILED", "CANCELLED"].includes(cloudStatus);
  return INSTALL_PHASES.map(([id, label, help]) => ({
    id,
    label,
    help,
    status: phaseState(id, job, localInstalled, cloudStatus),
  }));
}

export function extractInstallErrorCode(message) {
  if (!message) return "";
  const match = String(message).match(/\b([A-Z][A-Z0-9_]{2,})\b/);
  return match ? match[1] : "";
}

export function installationView(job) {
  const status = job?.status || "queued";
  const phases = installationPhases(job);
  const currentIndex = Math.max(0, phases.findIndex((phase) => ["UPLOADING", "RUNNING", "FAILED"].includes(phase.status)));
  const cloudStatus = cloudBackupStatus(job);
  const localInstalled = phases.slice(0, 3).every((phase) => phase.status === "COMPLETED");
  return {
    status,
    currentIndex,
    phases,
    cloudStatus,
    cloudUploading: cloudStatus === "UPLOADING",
    cloudCancelled: cloudStatus === "CANCELLED",
    localInstalled,
    terminal: ["completed", "failed", "cancelled"].includes(status) || (localInstalled && ["COMPLETED", "FAILED", "CANCELLED", "NOT_REQUESTED"].includes(cloudStatus)),
    complete: localInstalled,
    failed: status === "failed" && !localInstalled,
    failureCode: extractInstallErrorCode(job?.message),
    canRetry: status === "failed" && !localInstalled,
    cacheFailed: cloudStatus === "FAILED",
    canRetryCache: cloudStatus === "FAILED" && localInstalled,
  };
}

export function formatTransferBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  const gib = bytes / 1073741824;
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} Go`;
}

export function formatEta(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  const rest = rounded % 60;
  return minutes ? `${minutes} min ${rest} s` : `${rest} s`;
}

export function transferView(job) {
  const transfer = job?.transfer;
  if (!transfer || transfer.direction !== "upload") return null;
  const calculated = transfer.bytes_total > 0
    ? (transfer.bytes_transferred / transfer.bytes_total) * 100
    : 100;
  return {
    ...transfer,
    percent: Math.max(0, Math.min(100, calculated)),
    transferredLabel: formatTransferBytes(transfer.bytes_transferred),
    totalLabel: formatTransferBytes(transfer.bytes_total),
    rateLabel: transfer.bytes_per_second
      ? `${(transfer.bytes_per_second / 1048576).toFixed(1)} Mo/s`
      : "Mesure en cours",
    etaLabel: formatEta(transfer.eta_seconds),
  };
}

export function dependencyView(job) {
  const dependency = job?.dependency;
  if (!dependency?.import_name && !dependency?.package) return null;
  const status = String(dependency.status || "REQUIRED").toUpperCase();
  return {
    ...dependency,
    status,
    active: ["DOWNLOADING", "INSTALLING"].includes(status),
    installed: ["AVAILABLE", "INSTALLED"].includes(status),
    failed: status === "FAILED",
  };
}
