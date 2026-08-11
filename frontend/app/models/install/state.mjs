export const INSTALL_STEPS = [
  ["checking", "Vérification", "Compatibilité et espace disque"],
  ["restoring_cache", "Cache", "Recherche du snapshot S3"],
  ["downloading", "Téléchargement", "Octets reçus depuis Hugging Face"],
  ["validating_snapshot", "Validation", "Structure et poids du snapshot"],
  ["saving_cache", "Sauvegarde", "Publication du snapshot installé vers S3"],
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
  };
}
