export const INSTALL_STEPS = [
  ["checking", "Vérification", "Compatibilité et espace disque"],
  ["restoring_cache", "Cache", "Recherche du snapshot S3"],
  ["downloading", "Téléchargement", "Octets reçus depuis Hugging Face"],
  ["validating_snapshot", "Validation", "Structure et poids du snapshot"],
  ["resolving_dependencies", "Runtime", "Détection des imports requis"],
  ["installing_dependencies", "Dépendances", "Installation isolée et contrôlée"],
  ["saving_cache", "Sauvegarde", "Sauvegarde dans le cache S3"],
  ["installed", "Installé", "Le chargement runtime reste une action séparée"],
];

export function extractInstallErrorCode(message) {
  if (!message) return "";
  const match = String(message).match(/\b([A-Z][A-Z0-9_]{2,})\b/);
  return match ? match[1] : "";
}

export function installationView(job) {
  const status = job?.status || "queued";
  const currentIndex = Math.max(0, INSTALL_STEPS.findIndex(([stage]) => stage === job?.stage));
  return {
    status,
    currentIndex,
    terminal: ["completed", "failed", "cancelled"].includes(status),
    complete: status === "completed",
    failed: status === "failed",
    failureCode: extractInstallErrorCode(job?.message),
    canRetry: status === "failed",
    cacheFailed: job?.cache_status === "CACHE_FAILED",
    canRetryCache: job?.cache_status === "CACHE_FAILED",
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
