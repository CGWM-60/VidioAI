"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { BsArrowClockwise, BsCheckCircle, BsCloudArrowDown, BsCloudCheck } from "react-icons/bs";
import { apiFetch } from "../../lib/api";
import styles from "../../studio.module.css";

const STORED_JOBS = "vidioai.cloud.restore.jobs";

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  return `${(bytes / 1073741824).toFixed(1)} Go`;
}

export default function CloudModelsPage() {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiFetch("/api/models/cloud", { timeoutMs: 120000, timeoutCode: "CLOUD_LIST_TIMEOUT" });
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

  useEffect(() => {
    const saved = JSON.parse(window.localStorage.getItem(STORED_JOBS) || "[]");
    if (!saved.length) return undefined;
    let stopped = false;
    async function reconcile() {
      const current = await Promise.all(saved.map((id) => apiFetch(`/api/jobs/${id}`).catch(() => null)));
      if (stopped) return;
      const valid = current.filter(Boolean);
      setJobs(valid);
      const active = valid.filter((job) => !["completed", "failed", "cancelled"].includes(job.status));
      window.localStorage.setItem(STORED_JOBS, JSON.stringify(active.map((job) => job.id)));
      if (!active.length) void load();
    }
    void reconcile();
    const interval = window.setInterval(() => void reconcile(), 2000);
    return () => { stopped = true; window.clearInterval(interval); };
  }, [load]);

  const selectedModels = useMemo(() => models.filter((model) => selected.includes(`${model.repository}@${model.revision}`)), [models, selected]);
  const selectedBytes = useMemo(() => selectedModels.reduce((total, model) => total + model.size_bytes, 0), [selectedModels]);

  function toggle(model) {
    const identity = `${model.repository}@${model.revision}`;
    setSelected((values) => values.includes(identity) ? values.filter((value) => value !== identity) : [...values, identity]);
  }

  async function restore() {
    setError("");
    try {
      const created = await apiFetch("/api/models/cloud/restore", {
        method: "POST",
        timeoutMs: 120000,
        timeoutCode: "CLOUD_RESTORE_TIMEOUT",
        body: JSON.stringify({ models: selectedModels.map(({ repository, revision }) => ({ repository, revision })) }),
      });
      setJobs(created);
      window.localStorage.setItem(STORED_JOBS, JSON.stringify(created.map((job) => job.id)));
      setSelected([]);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div><h1>Sauvegardes cloud</h1><p>Snapshots S3 validés par manifeste, restaurables sans téléchargement Hugging Face.</p></div>
        <span className={styles.avatar}><BsCloudCheck /></span>
      </header>
      <div className={styles.catalogToolbar}>
        <div className={styles.stateCard}>{models.length} snapshot(s) valide(s)</div>
        <button className={styles.secondaryButton} onClick={() => void load()}><BsArrowClockwise /> Actualiser</button>
        <button className={styles.secondaryButton} disabled={!models.length} onClick={() => setSelected(selected.length === models.length ? [] : models.map((model) => `${model.repository}@${model.revision}`))}>{selected.length === models.length ? "Tout désélectionner" : "Tout sélectionner"}</button>
        <button className={styles.primaryButton} disabled={!selected.length} onClick={() => void restore()}><BsCloudArrowDown /> Restaurer ({selected.length})</button>
      </div>
      {!!selected.length && <div className={styles.stateCard}><strong>{selected.length} modèle(s) sélectionné(s)</strong> · {formatBytes(selectedBytes)} à restaurer</div>}
      {error && <div className={styles.errorBanner} role="alert"><strong>CLOUD_ERROR</strong> · {error}</div>}
      {jobs.map((job) => (
        <div className={styles.stateCard} key={job.id}>
          <strong>{job.model_id}</strong> · {job.stage} · {job.progress}%
          {job.transfer && <span> · {formatBytes(job.transfer.bytes_transferred)} / {formatBytes(job.transfer.bytes_total)}</span>}
          {job.error && <span> · {job.error.code}: {job.error.message}</span>}
        </div>
      ))}
      {loading ? <div className={styles.stateCard}>Lecture des manifestes S3…</div> : (
        <div className={styles.modelList}>
          {models.map((model) => {
            const identity = `${model.repository}@${model.revision}`;
            return (
              <article className={styles.modelRow} key={identity}>
                <label className={styles.modelArtwork}><input type="checkbox" checked={selected.includes(identity)} onChange={() => toggle(model)} /><BsCloudArrowDown /></label>
                <div className={styles.modelCopy}>
                  <div className={styles.modelTitleLine}><h2>{model.name}</h2><span className={styles.successPill}><BsCheckCircle /> {model.cloud_state}</span></div>
                  <div className={styles.modelMetadata}><span>{model.repository}</span><span>Révision {model.revision}</span><span>{formatBytes(model.size_bytes)}</span><span>{model.files} fichiers</span></div>
                  <div className={styles.capabilityList}>{model.capabilities.map((capability) => <span key={capability}>{capability.replaceAll("_", " ")}</span>)}</div>
                  <small>{model.manifest_uri}</small>
                </div>
                <div className={styles.modelSize}><span>État local</span><strong>{model.local_state}</strong></div>
                <button className={styles.secondaryButton} onClick={() => toggle(model)}>{selected.includes(identity) ? "Sélectionné" : "Sélectionner"}</button>
              </article>
            );
          })}
          {!models.length && <div className={styles.stateCard}>Aucun snapshot S3 complet et valide.</div>}
        </div>
      )}
    </div>
  );
}
