export const CATALOG_TIMEOUT_MS = 90_000;
export const MODEL_PREFLIGHT_TIMEOUT_MS = 120_000;

/**
 * L'API de liste HF ne vérifie pas l'accès aux fichiers gated/private.
 * `UNVERIFIED` doit donc rester actionnable : le backend fera le contrôle
 * définitif sur la fiche exacte avant de démarrer le téléchargement.
 */
export function accessStatus(model) {
  if (!model.gated && !model.private) return "AUTHORIZED";
  if (model.access_authorized) return "AUTHORIZED";
  return model.access_checked ? "ACCESS_REQUIRED" : "UNVERIFIED";
}
