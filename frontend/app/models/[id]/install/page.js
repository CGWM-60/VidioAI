"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { BsArrowLeft, BsCheck2, BsCircle, BsInfoCircle } from "react-icons/bs";
import { apiFetch, closeWebSocketSafely, eventsUrl } from "../../../lib/api";
import styles from "../../../studio.module.css";

const STEPS = [
  ["checking", "Vérification", "Compatibilité et espace disque"],
  ["restoring_cache", "Cache", "Recherche du snapshot S3"],
  ["downloading", "Téléchargement", "Octets reçus depuis Hugging Face"],
  ["validating_runtime", "Runtime", "Poids, chargement CUDA et inférence de test"],
  ["saving_cache", "Sauvegarde", "Publication du snapshot validé vers S3"],
  ["ready", "Prêt à l’emploi", "Démarrage du modèle"],
];

/** Suit le job par WebSocket après un unique GET d'amorçage. */
function InstallContent() {
  const { id } = useParams();
  const searchParams = useSearchParams();
  const modelId = searchParams.get("model_id") || decodeURIComponent(id || "");
  const jobId = searchParams.get("job");
  const [model, setModel] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const hasJob = Boolean(job);
  const terminal = ["completed", "failed", "cancelled"].includes(job?.status);

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
      if (envelope.event.startsWith("model.install") && envelope.data.id === jobId) setJob(envelope.data);
    };
    socket.onerror = () => {
      if (!intentionalClose) setError("Le canal temps réel est momentanément indisponible.");
    };
    return () => { intentionalClose = true; closeWebSocketSafely(socket); };
  }, [jobId, hasJob, terminal]);

  const currentIndex = useMemo(() => Math.max(0, STEPS.findIndex(([stage]) => stage === job?.stage)), [job]);
  const complete = job?.status === "completed";

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}><div className={styles.headingWithBack}><Link href="/models"><BsArrowLeft /></Link><div><h1>Installation automatique</h1><p>Téléchargement et installation simplifiés et automatisés.</p></div></div></header>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <section className={styles.installShell}>
        <div className={styles.installIntro}><h2>Installation de {model?.name || "votre modèle"}</h2><p>Vous pouvez suivre la progression réelle sans actualiser la page.</p></div>
        <div className={styles.installGrid}>
          <div className={styles.timeline}>
            {STEPS.map(([stage, label, help], index) => {
              const done = complete || index < currentIndex;
              const active = !complete && index === currentIndex;
              return <div className={`${styles.timelineStep} ${active ? styles.timelineActive : ""}`} key={`${stage}-${index}`}>
                <span>{done ? <BsCheck2 /> : <BsCircle />}</span><div><strong>{label}</strong><small>{active ? job?.message : help}</small></div>
              </div>;
            })}
          </div>
          <div className={styles.progressPanel}>
            <div className={styles.progressRing} style={{ "--progress": `${(job?.progress || 0) * 3.6}deg` }}><strong>{job?.progress || 0}<small>%</small></strong></div>
            <h2>{complete ? "Installation terminée" : job?.status === "failed" ? "Installation échouée" : "Installation en cours…"}</h2>
            <p>{job?.message || "Préparation du worker…"}</p>
            {complete && <Link className={styles.primaryButton} href={`/models/detail?model_id=${encodeURIComponent(modelId)}`}>Ouvrir le modèle</Link>}
            {job?.status === "failed" && <Link className={styles.secondaryButton} href="/models">Retour au catalogue</Link>}
          </div>
        </div>
        <div className={styles.tip}><BsInfoCircle /><div><strong>Astuce</strong><span>Le WebSocket vous notifie dès que le modèle est prêt à l’emploi.</span></div></div>
      </section>
    </div>
  );
}

// Next 16 demande une frontière Suspense autour de tout composant lisant les
// paramètres de recherche côté client.
export default function InstallPage() {
  return <Suspense fallback={<div className={styles.page}><div className={styles.stateCard}>Préparation du suivi…</div></div>}><InstallContent /></Suspense>;
}
