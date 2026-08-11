"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import {
  BsArrowClockwise, BsArrowRight, BsBox, BsChatDots, BsCheckCircle, BsCloudDownload,
  BsCpu, BsFilter, BsImage, BsSearch, BsStars,
} from "react-icons/bs";
import { apiFetch } from "../lib/api";
import styles from "../studio.module.css";

// Les filtres décrivent uniquement des catégories d'interface. Les modèles et
// leurs capacités restent exclusivement fournis par GET /api/models.
const FILTERS = [
  ["all", "Tous"], ["CHAT", "Chat"], ["IMAGE", "Image"],
  ["VIDEO", "Vidéo"], ["VISION", "Vision"], ["AUDIO", "Audio"],
  ["installed", "Installés"], ["compatible", "Compatibles"],
];

const CAPABILITY_LABELS = {
  CHAT: "Chat", TEXT_TO_IMAGE: "Texte → Image", IMAGE_TO_IMAGE: "Image → Image",
  INPAINTING: "Inpainting", OUTPAINTING: "Outpainting", IMAGE_VARIATION: "Variation d’image",
  IMAGE_UPSCALE: "Upscale image", CONTROLLED_IMAGE_GENERATION: "Génération contrôlée",
  TEXT_TO_VIDEO: "Texte → Vidéo", IMAGE_TO_VIDEO: "Image → Vidéo",
  MULTI_IMAGE_TO_VIDEO: "Multi-images → Vidéo", START_END_IMAGE_TO_VIDEO: "Start/End → Vidéo",
  KEYFRAMES_TO_VIDEO: "Keyframes → Vidéo", VIDEO_TO_VIDEO: "Vidéo → Vidéo",
  VIDEO_INPAINTING: "Inpainting vidéo", VIDEO_UPSCALE: "Upscale vidéo",
  AUDIO: "Audio", VISION: "Vision",
};

function ModelArtwork({ kind }) {
  const Icon = kind === "CHAT" ? BsChatDots : kind === "IMAGE" ? BsImage : BsStars;
  return <div className={`${styles.modelArtwork} ${styles[`art${kind}`]}`}><Icon /></div>;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "Taille inconnue";
  return `${(bytes / 1073741824).toFixed(bytes >= 10737418240 ? 0 : 1)} Go estimés`;
}

function formatMemory(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const gib = bytes / 1073741824;
  return `${gib.toFixed(gib >= 10 ? 0 : 1)} Go`;
}

function hardwareSummary(hardware) {
  if (!hardware || hardware.source === "unknown") return "Informations matérielles insuffisantes";
  if (hardware.source === "measured" && hardware.benchmark) {
    return `Mesuré sur ${hardware.benchmark.gpu} · VRAM pic ${formatMemory(hardware.benchmark.vram_peak_bytes)}`;
  }
  const source = hardware.source === "official" ? "Configuration officielle" : hardware.source === "partial" ? "Estimation partielle" : "Estimation matérielle";
  const range = Number.isFinite(hardware.estimated_vram_min) && Number.isFinite(hardware.estimated_vram_recommended)
    ? ` · VRAM ~${formatMemory(hardware.estimated_vram_min)}–${formatMemory(hardware.estimated_vram_recommended)}`
    : "";
  return `${source}${range}`;
}

function hardwareTooltip(hardware) {
  if (hardware?.source === "measured") return "Mesure enregistrée lors d’un chargement ou d’une génération réelle avec cette révision.";
  if (hardware?.source === "official") return "Exigence structurée publiée par l’auteur du modèle.";
  return "Estimation calculée à partir des poids et de la configuration Hugging Face. La consommation réelle peut varier selon la résolution et les paramètres de génération.";
}

function runtimeStatus(model) {
  return model.runtime_compatibility || (model.runtime_supported ? "SUPPORTED" : "UNSUPPORTED");
}

export default function ModelsPage() {
  const router = useRouter();
  const [models, setModels] = useState([]);
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState("");
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState("trending");
  const [meta, setMeta] = useState({ has_more: false, stale: false, last_sync: null, total: 0 });

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const parameters = new URLSearchParams({ page: String(page), limit: "20", sort });
      if (query.trim()) parameters.set("search", query.trim());
      if (["CHAT", "IMAGE", "VIDEO", "VISION", "AUDIO"].includes(filter)) parameters.set("category", filter);
      if (filter === "installed") parameters.set("installed", "true");
      if (filter === "compatible") parameters.set("compatible", "true");
      const response = await apiFetch(`/api/models?${parameters}`, { timeoutMs: 15000, timeoutCode: "CATALOG_TIMEOUT" });
      // La forme tableau reste tolérée durant un rolling deploy backend/frontend.
      setModels(Array.isArray(response) ? response : response.items);
      setMeta(Array.isArray(response) ? { has_more: false, stale: false, total: response.length } : response);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [filter, page, query, sort]);

  useEffect(() => {
    // Le callback de la promesse est asynchrone : l'effet ne provoque pas une
    // mise à jour d'état synchrone pendant le cycle de rendu React.
    // Un debounce court empêche une requête Hub à chaque caractère saisi.
    const timeout = window.setTimeout(() => { void loadModels(); }, 350);
    return () => window.clearTimeout(timeout);
  }, [loadModels]);

  function selectFilter(value) { setFilter(value); setPage(1); }

  async function refreshCatalog() {
    setLoading(true);
    try { await apiFetch("/api/models/catalog/refresh", { method: "POST" }); await loadModels(); }
    catch (requestError) { setError(requestError.message); setLoading(false); }
  }

  async function startInstall(model) {
    setBusyId(model.id);
    setError("");
    try {
      const job = await apiFetch("/api/models/install", {
        method: "POST",
        body: JSON.stringify({ model_id: model.id, revision: model.revision }),
      });
      router.push(`/models/install?model_id=${encodeURIComponent(model.id)}&job=${job.id}`);
    } catch (requestError) {
      setError(requestError.message);
      setBusyId("");
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div><h1>Catalogue de modèles</h1><p>Découvrez, téléchargez et installez des modèles en un clic.</p></div>
        <span className={styles.avatar}>A</span>
      </header>

      <div className={styles.catalogToolbar}>
        <label className={styles.searchBox}>
          <BsSearch aria-hidden="true" />
          <input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="Nom, auteur ou URL Hugging Face…" />
        </label>
        <div className={styles.filterTabs} aria-label="Filtrer les modèles">
          {FILTERS.map(([value, label]) => (
            <button key={value} className={filter === value ? styles.activeTab : ""} onClick={() => selectFilter(value)}>{label}</button>
          ))}
        </div>
        <label className={styles.catalogSort}><BsFilter /><select value={sort} onChange={(event) => { setSort(event.target.value); setPage(1); }}><option value="trending">Tendances</option><option value="downloads">Téléchargements</option><option value="likes">Likes</option><option value="updated">Récemment mis à jour</option><option value="name">Nom</option><option value="compatibility">Compatibilité VidioAI</option><option value="recommended">Recommandés</option></select></label>
        <button className={styles.secondaryButton} onClick={refreshCatalog}><BsArrowClockwise /> Actualiser</button>
      </div>

      {error && <div className={styles.errorBanner} role="alert"><strong>CATALOG_ERROR</strong> · {error} <button className={styles.secondaryButton} onClick={() => void loadModels()}>Réessayer</button></div>}
      {meta.stale && <div className={styles.warningBanner}>Hugging Face est momentanément indisponible : affichage du cache du {meta.last_sync ? new Date(meta.last_sync * 1000).toLocaleString("fr-FR") : "dernier accès"}.</div>}
      {loading ? <div className={styles.stateCard}>Chargement du catalogue réel…</div> : (
        <div className={styles.modelList}>
          {models.map((model) => (
            <article className={styles.modelRow} key={model.id}>
              <ModelArtwork kind={model.kind} />
              <div className={styles.modelCopy}>
                <div className={styles.modelTitleLine}>
                  <h2>{model.name}</h2>
                  {model.runtime_ready
                    ? <span className={styles.successPill}><BsCheckCircle /> Prêt</span>
                    : model.installed
                      ? <span className={styles.successPill}><BsCheckCircle /> Installé</span>
                      : null}
                </div>
                <div className={styles.modelMetadata}><span>{model.author || "Auteur inconnu"}</span><span>{formatBytes(model.estimated_size_bytes)}</span>{Number.isFinite(model.downloads) && <span>{model.downloads.toLocaleString("fr-FR")} téléchargements</span>}{Number.isFinite(model.likes) && <span>{model.likes.toLocaleString("fr-FR")} likes</span>}</div>
                <div className={styles.capabilityList}>
                  {model.capabilities.map((capability) => <span key={capability}>{CAPABILITY_LABELS[capability] || capability}</span>)}
                </div>
                <p>{model.description}</p>
                <small className={styles.hardwareSummary} title={hardwareTooltip(model.hardware)}><BsCpu /> {hardwareSummary(model.hardware)} · {model.compatibility_level.toLowerCase().replaceAll("_", " ")}</small>
                <div className={styles.compactCompatibility}>
                  <span className={model.hardware_compatible ? styles.checkGood : styles.checkBad}>{model.hardware_compatible ? "✓" : "✕"} Matériel</span>
                  <span className={runtimeStatus(model) === "SUPPORTED" ? styles.checkGood : runtimeStatus(model) === "UNKNOWN" ? styles.warningBanner : styles.checkBad}>{runtimeStatus(model) === "SUPPORTED" ? "✓ Pipeline runtime" : runtimeStatus(model) === "UNKNOWN" ? "? Validation après téléchargement" : "✕ Pipeline runtime"}</span>
                  <span className={model.source_available ? styles.checkGood : styles.checkBad}>{model.source_available ? "✓" : "✕"} Source</span>
                </div>
                {runtimeStatus(model) !== "SUPPORTED" && <small className={styles.runtimeReason}>{model.runtime_reason}</small>}
                {model.repository_url && <a className={styles.repositoryLink} href={model.repository_url} target="_blank" rel="noreferrer">Hugging Face · {model.repository}</a>}
              </div>
              <div className={styles.modelSize}>
                <span>Variante conseillée</span>
                <strong>{model.recommended_variant || "Aucune"}</strong>
              </div>
              <div className={styles.modelActions}>
                {!model.installed ? (
                  <button className={styles.primaryButton} title={!model.runtime_supported ? model.runtime_reason : model.gated && !model.access_authorized ? "Accès Hugging Face requis" : ""} disabled={!model.installable || busyId === model.id} onClick={() => startInstall(model)}>
                    <BsCloudDownload /> {busyId === model.id ? "Préparation…" : model.gated && !model.access_authorized ? "Accès requis" : runtimeStatus(model) === "SUPPORTED" ? "Installer" : runtimeStatus(model) === "UNKNOWN" ? "Valider et installer" : "Runtime non compatible"}
                  </button>
                ) : <span className={styles.readyBadge}>{model.runtime_ready ? "Prêt" : "Installé"}</span>}
                <Link href={`/models/detail?model_id=${encodeURIComponent(model.id)}`}>Détails <BsArrowRight /></Link>
              </div>
            </article>
          ))}
          {!models.length && <div className={styles.stateCard}><BsBox /> Aucun modèle ne correspond à ces filtres.</div>}
          <nav className={styles.catalogPagination} aria-label="Pagination du catalogue"><button className={styles.secondaryButton} disabled={page === 1} onClick={() => setPage((value) => value - 1)}>Précédent</button><span>Page {page}{meta.total ? ` · ${meta.total} résultats chargés` : ""}</span><button className={styles.secondaryButton} disabled={!meta.has_more} onClick={() => setPage((value) => value + 1)}>Suivant</button></nav>
        </div>
      )}
    </div>
  );
}
