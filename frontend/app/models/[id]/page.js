"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { BsArrowLeft, BsCheck2, BsCloudDownload, BsCpu, BsPlay, BsStop } from "react-icons/bs";
import { apiFetch } from "../../lib/api";
import styles from "../../studio.module.css";

function formatMemory(bytes) {
  if (!Number.isFinite(bytes)) return "Non disponible";
  const gib = bytes / 1073741824;
  return `${gib.toFixed(gib >= 10 ? 0 : 1)} Go`;
}

function formatRange(range) {
  if (!range || !Number.isFinite(range.min_bytes) || !Number.isFinite(range.max_bytes)) return "Non disponible";
  if (range.min_bytes === range.max_bytes) return formatMemory(range.min_bytes);
  return `~${formatMemory(range.min_bytes)}–${formatMemory(range.max_bytes)}`;
}

function sourceLabel(hardware) {
  if (hardware?.source === "measured") return `Mesuré sur ${hardware.benchmark?.gpu || "la machine actuelle"}`;
  if (hardware?.source === "official") return "Configuration officielle";
  if (hardware?.source === "estimated") return "Estimation matérielle";
  if (hardware?.source === "partial") return "Estimation partielle";
  return "Informations matérielles insuffisantes";
}

/** Page de détail entièrement alimentée par GET /api/models/{id}. */
export default function ModelDetailsPage() {
  const { id } = useParams();
  const router = useRouter();
  // Next conserve le slash encodé dans le segment dynamique. On normalise une
  // seule fois avant de construire l'URL API, sinon `%2F` devient `%252F`.
  const modelId = decodeURIComponent(id);
  const apiId = encodeURIComponent(modelId);
  const [model, setModel] = useState(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try { setModel(await apiFetch(`/api/models/${apiId}`)); setError(""); }
    catch (requestError) { setError(requestError.message); }
  }, [apiId]);

  useEffect(() => {
    const request = Promise.resolve().then(refresh);
    return () => { void request; };
  }, [refresh]);

  async function changeRuntime(action) {
    setRuntimeBusy(true);
    try { await apiFetch(`/api/models/${apiId}/${action}`, { method: "POST" }); await refresh(); }
    catch (requestError) { setError(requestError.message); }
    finally { setRuntimeBusy(false); }
  }

  async function startInstall() {
    setRuntimeBusy(true);
    try {
      const job = await apiFetch("/api/models/install", {
        method: "POST",
        body: JSON.stringify({ model_id: model.id, revision: model.revision }),
      });
      router.push(`/models/${encodeURIComponent(model.id)}/install?job=${job.id}`);
    } catch (requestError) { setError(requestError.message); setRuntimeBusy(false); }
  }

  if (error && !model) return <div className={styles.page}><div className={styles.errorBanner}>{error}</div></div>;
  if (!model) return <div className={styles.page}><div className={styles.stateCard}>Chargement du modèle…</div></div>;

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div className={styles.headingWithBack}><Link href="/models"><BsArrowLeft /></Link><div><h1>{model.name}</h1><p>{model.description}</p></div></div>
      </header>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <section className={`${styles.largePanel} ${styles.detailsGrid}`}>
        <div><span className={styles.eyebrow}>Moteur</span><h2>{model.engine}</h2></div>
        <div><span className={styles.eyebrow}>Licence</span><h2>{model.license}</h2></div>
        <div><span className={styles.eyebrow}>État local</span><h2>{model.loaded ? "Chargé" : model.installed ? "Installé" : "Absent"}</h2></div>
      </section>
      <section className={`${styles.largePanel} ${styles.detailsGrid}`}>
        <div><span className={styles.eyebrow}>Révision</span><h2>{model.revision.slice(0, 12)}</h2></div>
        <div><span className={styles.eyebrow}>Compatibilité</span><h2>{model.compatibility_level.replaceAll("_", " ")}</h2></div>
        <div><span className={styles.eyebrow}>Accès</span><h2>{model.accessibility.replaceAll("_", " ")}</h2></div>
      </section>
      {model.repository_url && <section className={styles.largePanel}><h2>Source officielle</h2><a className={styles.repositoryLink} href={model.repository_url} target="_blank" rel="noreferrer">{model.repository}</a></section>}
      <section className={styles.largePanel}>
        <h2>Capacités</h2>
        <div className={styles.capabilityCards}>{model.capabilities.map((item) => <span key={item}><BsCheck2 /> {item.replaceAll("_", " ")}</span>)}</div>
      </section>
      <section className={`${styles.largePanel} ${styles.hardwarePanel}`} title="Les estimations sont calculées à partir des poids et de la configuration Hugging Face. La consommation réelle varie selon la résolution et les paramètres de génération.">
        <div className={styles.hardwareHeading}>
          <div><span className={styles.eyebrow}>Configuration matérielle</span><h2>{sourceLabel(model.hardware)}</h2></div>
          <span className={styles.confidenceBadge}>Confiance {model.hardware.confidence === "high" ? "élevée" : model.hardware.confidence === "medium" ? "moyenne" : "faible"}</span>
        </div>
        {model.hardware.source !== "unknown" ? (
          <>
            <div className={styles.hardwareMetrics}>
              <div><span>VRAM d’inférence</span><strong>{model.hardware.source === "measured" ? formatMemory(model.hardware.benchmark?.vram_peak_bytes) : `~${formatMemory(model.hardware.estimated_vram_min)}–${formatMemory(model.hardware.estimated_vram_recommended)}`}</strong></div>
              <div><span>RAM</span><strong>{formatRange(model.hardware.estimated_ram)}</strong></div>
              <div><span>Poids seuls</span><strong>{formatRange(model.hardware.weights_memory)}</strong><small>Ce n’est pas la VRAM totale d’inférence.</small></div>
              <div><span>Backend</span><strong>{model.hardware.recommended_backend || "Non déterminé"}</strong></div>
              <div><span>Précision</span><strong>{model.hardware.recommended_precision || "Non déterminée"}</strong></div>
              <div><span>Compatibilité</span><strong>{model.hardware.compatibility_level.toLowerCase().replaceAll("_", " ")}</strong></div>
            </div>
            {model.hardware.benchmark?.inference_seconds && <p className={styles.measuredNote}>Temps mesuré : {model.hardware.benchmark.inference_seconds.toFixed(1)} s · batch {model.hardware.benchmark.batch}</p>}
            <ul className={styles.hardwareNotes}>{model.hardware.notes.map((note) => <li key={note}>{note}</li>)}</ul>
          </>
        ) : <p>Le repository ne fournit ni poids dimensionnés, ni métadonnées Safetensors, ni exigence officielle exploitable.</p>}
      </section>
      <section className={styles.largePanel}>
        <h2>Variantes</h2>
        <div className={styles.variantGrid}>{model.variants.map((variant) => (
          <div key={variant.id}><strong>{variant.label}</strong><span><BsCpu /> RAM {(variant.ram_required / 1073741824).toFixed(1)} Go · VRAM {(variant.vram_required / 1073741824).toFixed(1)} Go</span>{model.recommended_variant === variant.id && <em>Recommandée</em>}</div>
        ))}</div>
      </section>
      <div className={styles.footerActions}>
        {model.installed && !model.loaded && <button className={styles.primaryButton} disabled={runtimeBusy} onClick={() => changeRuntime("load")}><BsPlay /> Charger le modèle</button>}
        {model.loaded && <button className={styles.secondaryButton} disabled={runtimeBusy} onClick={() => changeRuntime("unload")}><BsStop /> Décharger</button>}
        {!model.installed && <button className={styles.primaryButton} disabled={!model.installable || runtimeBusy} onClick={startInstall}><BsCloudDownload /> {model.gated ? "Accès Hugging Face requis" : model.runtime_supported ? "Installer cette révision" : "Runtime non supporté"}</button>}
      </div>
    </div>
  );
}
