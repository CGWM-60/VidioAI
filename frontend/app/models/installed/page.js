"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { BsArrowClockwise, BsBox, BsCpu, BsGpuCard, BsPlayFill, BsStopFill } from "react-icons/bs";
import { apiFetch, closeWebSocketSafely, eventsUrl } from "../../lib/api";
import styles from "../../studio.module.css";
import { installedModelAction } from "./installed-state.mjs";

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  return `${(bytes / 1073741824).toFixed(1)} Go`;
}

function capabilityLabel(value) {
  return ({ TEXT_TO_IMAGE: "T2I", IMAGE_TO_IMAGE: "I2I", TEXT_TO_VIDEO: "T2V", IMAGE_TO_VIDEO: "I2V", VIDEO_TO_VIDEO: "V2V" })[value] || value.replaceAll("_", " ");
}

const STAGES = {
  resolving_precision: "Résolution précision…",
  planning_memory: "Planification mémoire…",
  loading_pipeline: "Chargement pipeline…",
  releasing_pipeline: "Libération du runtime…",
  memory_rejected: "VRAM/RAM insuffisante",
  ready: "READY",
};

export default function InstalledModelsPage() {
  const [payload, setPayload] = useState({ items: [], loaded: 0, gpu: null, memory: null });
  const [loading, setLoading] = useState(true);
  const [actions, setActions] = useState({});
  const [errors, setErrors] = useState({});
  const [focus] = useState(() => typeof window === "undefined" ? "" : new URLSearchParams(window.location.search).get("model") || "");

  const load = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    try {
      setPayload(await apiFetch("/api/models/installed", { timeoutMs: 30000, timeoutCode: "INSTALLED_MODELS_TIMEOUT" }));
    } catch (error) {
      setErrors((current) => ({ ...current, page: error.message }));
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const request = Promise.resolve().then(() => load());
    const interval = window.setInterval(() => void load({ quiet: true }), 2500);
    return () => { void request; window.clearInterval(interval); };
  }, [load]);

  useEffect(() => {
    const socket = new WebSocket(eventsUrl());
    socket.addEventListener("message", (event) => {
      try {
        const envelope = JSON.parse(event.data);
        if (envelope.event === "resources.updated") void load({ quiet: true });
      } catch {
        // GET périodique authoritative.
      }
    });
    return () => closeWebSocketSafely(socket);
  }, [load]);

  async function act(model) {
    const action = installedModelAction(model);
    setActions((current) => ({ ...current, [model.id]: action }));
    setErrors((current) => ({ ...current, [model.id]: "" }));
    try {
      await apiFetch(`/api/models/${action}`, {
        method: "POST",
        timeoutMs: action === "load" ? 1800000 : 60000,
        timeoutCode: action === "load" ? "MODEL_LOAD_TIMEOUT" : "MODEL_UNLOAD_TIMEOUT",
        body: JSON.stringify({ model_id: model.id }),
      });
      await load({ quiet: true });
    } catch (error) {
      setErrors((current) => ({ ...current, [model.id]: error.message }));
      await load({ quiet: true });
    } finally {
      setActions((current) => ({ ...current, [model.id]: "" }));
    }
  }

  const gpu = payload.gpu || {};
  const memory = payload.memory || {};
  const ramUsed = Math.max(0, Number(memory.ram_total_bytes || 0) - Number(memory.ram_available_bytes || 0));
  const vramFree = Math.max(0, Number(gpu.vram_total_bytes || 0) - Number(gpu.vram_used_bytes || 0));
  const items = useMemo(() => [...(payload.items || [])].sort((left, right) => Number(right.id === focus) - Number(left.id === focus)), [focus, payload.items]);

  return (
    <div className={styles.page}>
      <nav className={styles.modelNavigation} aria-label="Navigation modèles">
        <Link href="/models">Catalogue</Link><Link className={styles.modelNavigationActive} href="/models/installed">Installés</Link><Link href="/models/cloud">Sauvegardes cloud</Link>
      </nav>
      <header className={styles.pageHeading}>
        <div><h1><BsBox /> Modèles installés</h1><p>Modèles disponibles localement sur le Scratch</p></div>
        <button className={styles.secondaryButton} disabled={loading} onClick={() => void load()}><BsArrowClockwise /> Actualiser</button>
      </header>

      <div className={styles.installedSummary}>
        <div><strong>{items.length}</strong><span>modèles installés</span></div>
        <div><strong>{payload.loaded || 0}</strong><span>chargés</span></div>
        <div><BsGpuCard /><strong>{formatBytes(gpu.vram_used_bytes)} / {formatBytes(gpu.vram_total_bytes)}</strong><span>VRAM</span></div>
        <div><BsCpu /><strong>{formatBytes(ramUsed)} / {formatBytes(memory.ram_total_bytes)}</strong><span>RAM</span></div>
      </div>

      {errors.page && <div className={styles.errorBanner}><strong>RUNTIME_ERROR</strong> · {errors.page}<button className={styles.secondaryButton} onClick={() => void load()}>Réessayer</button></div>}
      {loading ? <div className={styles.stateCard}>Lecture des modèles présents sur le Scratch…</div> : items.length ? (
        <div className={styles.installedGrid}>
          {items.map((model) => {
            const action = installedModelAction(model);
            const busy = actions[model.id] || ["LOADING", "UNLOADING"].includes(model.state);
            const plan = model.memory_plan || {};
            return (
              <article className={`${styles.installedCard} ${model.id === focus ? styles.installedCardFocused : ""}`} id={`model-${model.storage_id}`} key={model.storage_id}>
                <div className={styles.installedCardHeading}><div><h2>{model.repository}</h2><span className={model.loaded ? styles.successPill : styles.neutralPill}>{model.state}</span></div><small>{STAGES[model.stage] || model.stage || (model.loaded ? "Pipeline confirmé par le Worker" : "Disponible localement")}</small></div>
                <dl className={styles.installedDetails}>
                  <div><dt>Capacités</dt><dd>{model.capabilities.map(capabilityLabel).join(" · ") || "—"}</dd></div>
                  <div><dt>Précision</dt><dd>{model.precision}</dd></div>
                  <div><dt>Pipeline</dt><dd>{model.pipeline_class || "À résoudre au chargement"}</dd></div>
                  <div><dt>Stockage</dt><dd>{formatBytes(model.size_bytes)}</dd></div>
                  <div><dt>Révision</dt><dd>{model.revision.slice(0, 12)}…</dd></div>
                  <div><dt>Runtime</dt><dd>{model.device || (plan.strategy === "FULL_GPU" ? "GPU" : plan.strategy || "AUTO")}</dd></div>
                  <div><dt>VRAM</dt><dd>{formatBytes(model.vram_bytes)} · pic {formatBytes(model.vram_peak_bytes)}</dd></div>
                  <div><dt>VRAM machine</dt><dd>{formatBytes(gpu.vram_used_bytes)} / {formatBytes(gpu.vram_total_bytes)} · libre {formatBytes(vramFree)}</dd></div>
                  <div><dt>RAM</dt><dd>disponible {formatBytes(memory.ram_available_bytes)} / {formatBytes(memory.ram_total_bytes)} · VidioAI {formatBytes(memory.vidioai_ram_bytes)} · pic {formatBytes(model.ram_peak_bytes)}</dd></div>
                  <div><dt>Scratch disponible</dt><dd>{formatBytes(memory.scratch_available_bytes)}</dd></div>
                  <div><dt>Stratégie</dt><dd>{model.memory_strategy || plan.strategy || "À planifier"}</dd></div>
                  <div><dt>Memory plan</dt><dd>poids {formatBytes(plan.model_bytes)} · headroom {formatBytes(plan.inference_headroom_bytes)} · pipeline {formatBytes(plan.vram_pipeline_bytes)}</dd></div>
                  <div><dt>Offload CPU</dt><dd>{model.cpu_offload ? "Oui" : "Non"}</dd></div>
                  <div><dt>Offload disque</dt><dd>{model.disk_offload ? "Oui" : "Non"}</dd></div>
                </dl>
                {(errors[model.id] || model.error) && <div className={styles.inlineRuntimeError}>{errors[model.id] || model.error}</div>}
                <button className={action === "unload" ? styles.secondaryButton : styles.primaryButton} disabled={Boolean(busy)} onClick={() => void act(model)}>{action === "unload" ? <><BsStopFill /> {busy ? "Déchargement…" : "Décharger"}</> : <><BsPlayFill /> {busy ? "Chargement…" : "Charger"}</>}</button>
              </article>
            );
          })}
        </div>
      ) : <div className={styles.stateCard}>Aucun modèle installé sur le Scratch.</div>}
    </div>
  );
}
