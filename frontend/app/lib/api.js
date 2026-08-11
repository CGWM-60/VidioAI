// Point d'entrée HTTP unique du frontend. Le remplacement par une URL de
// production se fait avec NEXT_PUBLIC_API_URL, sans toucher aux pages.
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

// Le WebSocket utilise le même hôte que l'API et adapte automatiquement le
// protocole HTTP/HTTPS vers WS/WSS.
export function eventsUrl() {
  const configured = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "");
  return `${configured.replace(/^http/, "ws")}/api/events`;
}

/** URL du flux conversationnel, également relative à l'origine courante.
 * Elle fonctionne ainsi derrière Nginx, en HTTPS et sur un autre domaine. */
export function chatEventsUrl() {
  const configured = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "");
  return `${configured.replace(/^http/, "ws")}/api/chat/stream`;
}

/** Ferme un WebSocket sans provoquer l'avertissement Chromium émis lorsqu'un
 * composant React est démonté pendant la phase CONNECTING. */
export function closeWebSocketSafely(socket) {
  if (!socket) return;
  if (socket.readyState === WebSocket.OPEN) {
    socket.close();
  } else if (socket.readyState === WebSocket.CONNECTING) {
    socket.addEventListener("open", () => socket.close(), { once: true });
  }
}

/**
 * Exécute une requête JSON et transforme toutes les erreurs backend en Error.
 * Les pages n'ont ainsi qu'un seul chemin d'affichage pour les échecs réseau et
 * les validations métier renvoyées par Rust.
 */
export async function apiFetch(path, options = {}) {
  const { timeoutMs = 30000, timeoutCode = "REQUEST_TIMEOUT", ...fetchOptions } = options;
  const controller = !fetchOptions.signal && timeoutMs > 0 ? new AbortController() : null;
  const timeout = controller
    ? globalThis.setTimeout(() => controller.abort("VIDIOAI_REQUEST_TIMEOUT"), timeoutMs)
    : null;

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
      ...fetchOptions,
      signal: controller?.signal || fetchOptions.signal,
      headers: {
        ...(fetchOptions.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...fetchOptions.headers,
      },
    });
  } catch (error) {
    if (controller?.signal.aborted) {
      throw new Error(`${timeoutCode}: délai de ${Math.round(timeoutMs / 1000)} s dépassé.`);
    }
    throw error;
  } finally {
    if (timeout) globalThis.clearTimeout(timeout);
  }

  if (!response.ok) {
    let message = `Le serveur a répondu avec le statut ${response.status}.`;
    try {
      const payload = await response.json();
      message = payload.error
        ? `${payload.code ? `${payload.code}: ` : ""}${payload.error}`
        : message;
    } catch {
      // Une réponse non JSON conserve le message HTTP générique ci-dessus.
    }
    throw new Error(message);
  }

  return response.status === 204 ? null : response.json();
}

/** Construit l'URL publique d'un asset retourné par le backend. */
export function assetUrl(assetId) {
  return assetId ? `${API_BASE_URL}/api/assets/${assetId}` : "";
}
