"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { BsArrowLeft, BsCheck2, BsCircle, BsInfoCircle, BsXCircle } from "react-icons/bs";
import { apiFetch, closeWebSocketSafely, eventsUrl } from "../../../lib/api";
import { ProgressBar } from "../../../components/ui";
import {
  cancelCloudBackupPayload,
  dependencyView,
  installationView,
  transferView,
} from "../../install/state.mjs";
import styles from "../../../studio.module.css";

/** Suit le job par WebSocket après un unique GET d'amorçage. */
function InstallContent() {
  const { id } = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const modelId = searchParams.get("model_id") || decodeURIComponent(id || "");
  const fallbackModelName = decodeURIComponent(modelId || "").split("/").filter(Boolean).pop() || "Modèle inconnu";
  const jobId = searchParams.get("job");
  const [model, setModel] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [retrying, setRetrying] = useState(false);
  const [cancellingCloud, setCancellingCloud] = useState(false);
  const hasJob = Boolean(job);
  const view = installationView(job);
  const { terminal, failureCode, complete } = view;
  const transfer = transferView(job);
  const dependency = dependencyView(job);

  const loadInitialState = useCallback(async () => {
    if (!jobId) { setError("Identifiant de job absent."); return; }
    try {
      const [modelData, jobData] = await Promise.all([
        apiFetch(`/api/models/by-id?model_id=${encodeURIComponent(modelId)}`), apiFetch(`/api/jobs/${jobId}`),
      ]);
      setModel(modelData); setJob(jobData);
    } catch (requestError) { setError(requestError.message); }
  }, [jobId, modelId]);

  useEffect(() => {
    const request = Promise.resolve().then(loadInitialState);
    return () => { void request; };
  }, [loadInitialState]);

  useEffect(() => {
    // Le GET initial peut déjà révéler un job terminé. Dans ce cas ouvrir puis
    // fermer immédiatement un WebSocket produirait une fausse erreur visuelle.
    if (!jobId || !hasJob || terminal) return undefined;
    let intentionalClose = false;
    const socket = new WebSocket(eventsUrl());
    socket.onmessage = (message) => {
      const envelope = JSON.parse(message.data);
      if ((envelope.event.startsWith("model.install") || envelope.event.startsWith("model.cache") || envelope.event.startsWith("model.dependency")) && envelope.data.id === jobId) setJob(envelope.data);
    };
    socket.onerror = () => {
      if (!intentionalClose) setError("Le canal temps réel est momentanément indisponible.");
    };
    return () => { intentionalClose = true; closeWebSocketSafely(socket); };
  }, [jobId, hasJob, terminal]);

  const currentIndex = useMemo(() => installationView(job).currentIndex, [job]);

  async function retryInstall() {
    if (!model?.id) return;
    setRetrying(true);
    try {
      const nextJob = await apiFetch("/api/models/install", {
        method: "POST",
        body: JSON.stringify({ model_id: model.id, revision: model.revision }),
      });
      router.push(`/models/install?model_id=${encodeURIComponent(model.id)}&job=${nextJob.id}`);
    } catch (requestError) {
      setError(requestError.message);
      setRetrying(false);
    }
  }

  async function retryCache() {
    if (!model?.id) return;
    setRetrying(true);
    try {
      const nextJob = await apiFetch("/api/models/cache", {
        method: "POST",
        body: JSON.stringify({ model_id: model.id }),
      });
      router.push(`/models/install?model_id=${encodeURIComponent(model.id)}&job=${nextJob.id}`);
    } catch (requestError) {
      setError(requestError.message);
      setRetrying(false);
    }
  }

  async function cancelCloudBackup() {
    if (!modelId || cancellingCloud) return;
    setCancellingCloud(true);
    setError("");
    try {
      const response = await apiFetch("/api/models/cloud-backup/cancel", {
        method: "POST",
        body: JSON.stringify(cancelCloudBackupPayload(modelId)),
      });
      setJob((current) => ({
        ...current,
        ...(response?.job || (response?.id ? response : {})),
        local_installation_status: response?.local_installation_status || current?.local_installation_status || "COMPLETED",
        cloud_backup_status: response?.cloud_backup_status || "CANCELLED",
        cache_status: response?.cache_status || "CACHE_CANCELLED",
        status: response?.status || "completed",
        stage: response?.stage || "installed",
        message: response?.message || "Modèle installé localement · sauvegarde cloud annulée.",
      }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setCancellingCloud(false);
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}><div className={styles.headingWithBack}><Link href="/models"><BsArrowLeft /></Link><div><h1>Installation automatique</h1><p>Téléchargement et installation simplifiés et automatisés.</p></div></div></header>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <section className={styles.installShell}>
        <div className={styles.installIntro}><h2>Installation de {model?.name || fallbackModelName}</h2><p>{model?.repository || modelId} · Vous pouvez suivre la progression réelle sans actualiser la page.</p></div>
        <div className={styles.installGrid}>
          <div className={styles.timeline}>
            {view.phases.map((phase, index) => {
              const done = ["COMPLETED", "SKIPPED"].includes(phase.status);
              const active = ["UPLOADING", "RUNNING"].includes(phase.status) || index === currentIndex && !view.terminal;
              return <div className={`${styles.timelineStep} ${active ? styles.timelineActive : ""}`} key={phase.id}>
                <span>{done ? <BsCheck2 /> : phase.status === "FAILED" ? <BsXCircle /> : <BsCircle />}</span>
                <div><strong>{phase.label}</strong><small>{active ? job?.message : phase.status === "CANCELLED" ? "Sauvegarde cloud annulée; installation locale conservée" : phase.help}</small></div>
              </div>;
            })}
          </div>
          <div className={styles.progressPanel}>
            <div className={styles.progressRing} style={{ "--progress": `${(job?.progress || 0) * 3.6}deg` }}><strong>{job?.progress || 0}<small>%</small></strong></div>
            <h2>{view.cloudCancelled ? "Modèle installé localement" : complete ? "Installation terminée" : view.failed ? `Installation échouée à ${job?.progress || 0}%` : "Installation en cours…"}</h2>
            <p>{job?.message || "Préparation du worker…"}</p>
            {view.cloudCancelled && <div className={styles.successBanner}><BsCheck2 /><span>Sauvegarde cloud annulée. Le modèle local reste installé et utilisable.</span></div>}
            {dependency && (
              <div className={styles.dependencyProgress}>
                <strong>{dependency.package || dependency.import_name}</strong>
                <span>{dependency.version || "version contrôlée"} · {dependency.status}</span>
              </div>
            )}
            {transfer && (
              <div className={styles.transferProgress}>
                <ProgressBar value={transfer.percent} label="Cache S3" showValue />
                <strong>{transfer.current_file || "Validation du manifeste"}</strong>
                <span>{transfer.transferredLabel} / {transfer.totalLabel}</span>
                <span>Débit : {transfer.rateLabel}</span>
                <span>Restant estimé : {transfer.etaLabel}</span>
                {transfer.files_skipped > 0 && <span>{transfer.files_skipped} fichier(s) déjà présent(s) dans le cache S3</span>}
                {view.cloudUploading && <button className={styles.dangerButton} disabled={cancellingCloud} onClick={() => void cancelCloudBackup()}><BsXCircle /> {cancellingCloud ? "Annulation…" : "Annuler la sauvegarde cloud"}</button>}
              </div>
            )}
            {view.cloudUploading && !transfer && <button className={styles.dangerButton} disabled={cancellingCloud} onClick={() => void cancelCloudBackup()}><BsXCircle /> {cancellingCloud ? "Annulation…" : "Annuler la sauvegarde cloud"}</button>}
            {job?.status === "failed" && failureCode && (
              <p className={styles.failureCode}>Code: {failureCode}</p>
            )}
            {complete && <Link className={styles.primaryButton} href={`/models/detail?model_id=${encodeURIComponent(modelId)}`}>Ouvrir et charger le modèle</Link>}
            {complete && view.canRetryCache && <button className={styles.secondaryButton} disabled={retrying} onClick={retryCache}>{retrying ? "Relance…" : "Réessayer la sauvegarde S3"}</button>}
            {view.failed && (
              <div className={styles.installFailureActions}>
                <button className={styles.primaryButton} disabled={retrying} onClick={retryInstall}>
                  {retrying ? "Relance…" : "Réessayer"}
                </button>
                <Link className={styles.secondaryButton} href="/models">Retour au catalogue</Link>
              </div>
            )}
          </div>
        </div>
        <div className={styles.tip}><BsInfoCircle /><div><strong>Étapes séparées</strong><span>L’installation ne lance ni CUDA ni inférence. Chargez ensuite explicitement le modèle depuis sa fiche.</span></div></div>
      </section>
    </div>
  );
}

// Next 16 demande une frontière Suspense autour de tout composant lisant les
// paramètres de recherche côté client.
export default function InstallPage() {
  return <Suspense fallback={<div className={styles.page}><div className={styles.stateCard}>Préparation du suivi…</div></div>}><InstallContent /></Suspense>;
}
