"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BsArrowClockwise, BsCheckCircle, BsCloudArrowDown, BsCloudCheck,
  BsExclamationTriangle, BsHddStack,
} from "react-icons/bs";
import { apiFetch, closeWebSocketSafely, eventsUrl } from "../../lib/api";
import styles from "../../studio.module.css";
import { CLOUD_TERMINAL_STATUSES, cloudJobPresentation, cloudRestorePayload, restoredModelHref } from "./cloud-state.mjs";

const STORED_JOBS = "vidioai.cloud.restore.jobs";

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  return `${(bytes / 1073741824).toFixed(bytes >= 10737418240 ? 1 : 2)} Go`;
}

function formatRate(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  return `${(bytes / 1048576).toFixed(1)} Mo/s`;
}

function capabilityLabel(value) {
  return ({ TEXT_TO_IMAGE: "T2I", IMAGE_TO_IMAGE: "I2I", TEXT_TO_VIDEO: "T2V", IMAGE_TO_VIDEO: "I2V", VIDEO_TO_VIDEO: "V2V" })[value] || value.replaceAll("_", " ");
}

export default function CloudModelsPage() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [trackedIds, setTrackedIds] = useState(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(window.localStorage.getItem(STORED_JOBS) || "[]"); }
    catch { return []; }
  });
  const [loading, setLoading] = useState(true);
  const [restoring, setRestoring] = useState(false);
  const [error, setError] = useState("");
  const observedTerminals = useRef(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiFetch("/api/models/cloud", { timeoutMs: 30000, timeoutCode: "CLOUD_LIST_TIMEOUT" });
      setModels(response.items || []);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const request = Promise.resolve().then(load);
    return () => { void request; };
  }, [load]);

  const rememberJobs = useCallback((created) => {
    setJobs((current) => {
      const byId = new Map(current.map((job) => [job.id, job]));
      created.forEach((job) => byId.set(job.id, job));
      return [...byId.values()];
    });
    setTrackedIds((current) => {
      const ids = [...new Set([...current, ...created.map((job) => job.id)])].slice(-20);
      window.localStorage.setItem(STORED_JOBS, JSON.stringify(ids));
      return ids;
    });
  }, []);

  useEffect(() => {
    if (!trackedIds.length) return undefined;
    let stopped = false;
    async function reconcile() {
      const current = await Promise.all(trackedIds.map((id) => apiFetch(`/api/jobs/${id}`).catch(() => null)));
      if (stopped) return;
      const valid = current.filter(Boolean);
      setJobs(valid);
      const newTerminal = valid.some((job) => {
        if (!CLOUD_TERMINAL_STATUSES.has(job.status) || observedTerminals.current.has(job.id)) return false;
        observedTerminals.current.add(job.id);
        return true;
      });
      if (newTerminal) void load();
    }
    void reconcile();
    const interval = window.setInterval(() => void reconcile(), 2000);
    return () => { stopped = true; window.clearInterval(interval); };
  }, [load, trackedIds]);

  useEffect(() => {
    const socket = new WebSocket(eventsUrl());
    socket.addEventListener("message", (event) => {
      try {
        const envelope = JSON.parse(event.data);
        if (envelope.event !== "job.updated" || !trackedIds.includes(envelope.data?.id)) return;
        setJobs((current) => {
          const byId = new Map(current.map((job) => [job.id, job]));
          byId.set(envelope.data.id, envelope.data);
          return [...byId.values()];
        });
        if (CLOUD_TERMINAL_STATUSES.has(envelope.data.status)) void load();
      } catch {
        // Le polling GET reste authoritative.
      }
    });
    return () => closeWebSocketSafely(socket);
  }, [load, trackedIds]);

  const selectedModels = useMemo(() => models.filter((model) => selected.includes(`${model.repository}@${model.revision}`)), [models, selected]);
  const selectedBytes = useMemo(() => selectedModels.reduce((total, model) => total + model.size_bytes, 0), [selectedModels]);
  const totalBytes = useMemo(() => models.reduce((total, model) => total + model.size_bytes, 0), [models]);
  const localCount = models.filter((model) => model.local).length;

  function toggle(model) {
    const identity = `${model.repository}@${model.revision}`;
    setSelected((values) => values.includes(identity) ? values.filter((value) => value !== identity) : [...values, identity]);
  }

  async function restore(targets = selectedModels) {
    if (!targets.length || restoring) return;
    setRestoring(true);
    setError("");
    try {
      const created = await apiFetch("/api/models/cloud/restore", {
        method: "POST",
        timeoutMs: 30000,
        timeoutCode: "CLOUD_RESTORE_TIMEOUT",
        body: JSON.stringify(cloudRestorePayload(targets)),
      });
      rememberJobs(created);
      setSelected([]);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setRestoring(false);
    }
  }

  return (
    <div className={styles.page}>
      <nav className={styles.modelNavigation} aria-label="Navigation modèles">
        <Link href="/models">Catalogue</Link><Link href="/models/installed">Installés</Link><Link className={styles.modelNavigationActive} href="/models/cloud">Sauvegardes cloud</Link>
      </nav>
      <header className={styles.pageHeading}>
        <div><h1><BsCloudCheck /> Sauvegardes cloud</h1><p>Modèles sauvegardés dans Scaleway Object Storage</p></div>
        <div className={styles.headingActions}>
          <button className={styles.secondaryButton} disabled={loading} onClick={() => void load()}><BsArrowClockwise /> Actualiser</button>
          <button className={styles.secondaryButton} disabled={!models.length} onClick={() => setSelected(selected.length === models.length ? [] : models.map((model) => `${model.repository}@${model.revision}`))}>Tout sélectionner</button>
          <button className={styles.primaryButton} disabled={!selected.length || restoring} onClick={() => void restore()}><BsCloudArrowDown /> Restaurer la sélection</button>
        </div>
      </header>

      <div className={styles.cloudSummary}>
        <div><strong>{models.length}</strong><span>sauvegardes</span></div>
        <div><strong>{formatBytes(totalBytes)}</strong><span>disponibles</span></div>
        <div><strong>{localCount} / {models.length}</strong><span>modèles locaux</span></div>
        {!!selected.length && <div><strong>{selected.length} · {formatBytes(selectedBytes)}</strong><span>sélectionnés</span></div>}
      </div>

      {error && <div className={styles.errorBanner} role="alert"><BsExclamationTriangle /><span><strong>CLOUD_ERROR</strong> · {error}</span><button className={styles.secondaryButton} onClick={() => void load()}>Réessayer</button></div>}

      {!!jobs.length && <section className={styles.restoreJobs}>
        {jobs.map((job) => {
          const view = cloudJobPresentation(job);
          const installedHref = restoredModelHref(job);
          const transfer = job.transfer;
          return (
            <article className={`${styles.restoreJob} ${view.failed ? styles.restoreJobFailed : view.completed ? styles.restoreJobCompleted : ""}`} key={job.id}>
              <div className={styles.restoreJobHeading}>
                <div><strong>{job.model_id}</strong><span>{view.failed ? "Échec de restauration" : view.completed ? "Restauré" : "Restauration depuis S3"}</span></div>
                {!view.failed && <strong>{view.progress}%</strong>}
              </div>
              {view.restoring && <><div className={styles.restoreProgress}><span style={{ width: `${view.progress}%` }} /></div><div className={styles.restoreMetrics}><span>{formatBytes(transfer?.bytes_transferred || 0)} / {formatBytes(transfer?.bytes_total || 0)}</span><span>{formatRate(transfer?.bytes_per_second)}</span><span>{transfer?.current_file || job.stage}</span></div></>}
              {view.failed && <div className={styles.restoreError}><strong>{view.errorCode}</strong><span>{view.errorMessage}</span><button className={styles.secondaryButton} onClick={() => void restore(models.filter((model) => model.repository === job.model_id))}>Réessayer</button></div>}
              {installedHref && <Link className={styles.secondaryButton} href={installedHref}>Voir le modèle</Link>}
            </article>
          );
        })}
      </section>}

      {loading ? <div className={styles.stateCard}>Lecture des manifestes S3…</div> : models.length ? (
        <div className={styles.cloudGrid}>
          {models.map((model) => {
            const identity = `${model.repository}@${model.revision}`;
            const active = jobs.some((job) => job.model_id === model.repository && !CLOUD_TERMINAL_STATUSES.has(job.status));
            return (
              <article className={`${styles.cloudCard} ${selected.includes(identity) ? styles.cloudCardSelected : ""}`} key={identity}>
                <div className={styles.cloudCardHeading}><label><input type="checkbox" checked={selected.includes(identity)} onChange={() => toggle(model)} /><span>{model.repository}</span></label><BsHddStack /></div>
                <dl className={styles.compactDetails}><div><dt>Cloud</dt><dd>Disponible</dd></div><div><dt>Local</dt><dd>{model.local_state === "ABSENT" ? "Absent" : "Installé"}</dd></div><div><dt>Taille</dt><dd>{formatBytes(model.size_bytes)}</dd></div><div><dt>Révision</dt><dd>{model.revision.slice(0, 10)}…</dd></div><div><dt>Capacités</dt><dd>{model.capabilities.map(capabilityLabel).join(" · ") || "—"}</dd></div></dl>
                {model.local_state !== "ABSENT" ? <Link className={styles.secondaryButton} href={`/models/installed?model=${encodeURIComponent(model.repository)}`}><BsCheckCircle /> Voir le modèle</Link> : <button className={styles.secondaryButton} disabled={active || restoring} onClick={() => void restore([model])}>{active ? "Restauration…" : "Restaurer"}</button>}
              </article>
            );
          })}
        </div>
      ) : <div className={styles.stateCard}>Aucune sauvegarde complète disponible.</div>}
    </div>
  );
}
