"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { BsArrowLeft, BsCheck2, BsCircle, BsInfoCircle } from "react-icons/bs";
import { apiFetch, closeWebSocketSafely, eventsUrl } from "../../../lib/api";
import { ProgressBar } from "../../../components/ui";
import { dependencyView, INSTALL_STEPS, installationView, transferView } from "../../install/state.mjs";
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

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}><div className={styles.headingWithBack}><Link href="/models"><BsArrowLeft /></Link><div><h1>Installation automatique</h1><p>Téléchargement et installation simplifiés et automatisés.</p></div></div></header>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <section className={styles.installShell}>
        <div className={styles.installIntro}><h2>Installation de {model?.name || fallbackModelName}</h2><p>{model?.repository || modelId} · Vous pouvez suivre la progression réelle sans actualiser la page.</p></div>
        <div className={styles.installGrid}>
          <div className={styles.timeline}>
            {INSTALL_STEPS.map(([stage, label, help], index) => {
              const done = complete || index < currentIndex;
              const active = !complete && index === currentIndex;
              return <div className={`${styles.timelineStep} ${active ? styles.timelineActive : ""}`} key={`${stage}-${index}`}>
                <span>{done ? <BsCheck2 /> : <BsCircle />}</span><div><strong>{label}</strong><small>{active ? job?.message : help}</small></div>
              </div>;
            })}
          </div>
          <div className={styles.progressPanel}>
            <div className={styles.progressRing} style={{ "--progress": `${(job?.progress || 0) * 3.6}deg` }}><strong>{job?.progress || 0}<small>%</small></strong></div>
            <h2>{complete ? "Installation terminée" : job?.status === "failed" ? `Installation échouée à ${job?.progress || 0}%` : "Installation en cours…"}</h2>
            <p>{job?.message || "Préparation du worker…"}</p>
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
              </div>
            )}
            {job?.status === "failed" && failureCode && (
              <p className={styles.failureCode}>Code: {failureCode}</p>
            )}
            {complete && <Link className={styles.primaryButton} href={`/models/detail?model_id=${encodeURIComponent(modelId)}`}>Ouvrir et charger le modèle</Link>}
            {complete && view.canRetryCache && <button className={styles.secondaryButton} disabled={retrying} onClick={retryCache}>{retrying ? "Relance…" : "Réessayer la sauvegarde S3"}</button>}
            {job?.status === "failed" && (
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
