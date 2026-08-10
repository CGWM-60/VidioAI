"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { BsArrowRight, BsCameraVideo, BsDatabase, BsImage, BsLightning, BsPlus, BsStars } from "react-icons/bs";
import { apiFetch } from "./lib/api";
import styles from "./studio.module.css";

export default function Home() {
  const [dashboard, setDashboard] = useState(null);
  const [models, setModels] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const request = Promise.all([apiFetch("/api/dashboard"), apiFetch("/api/models?installed=true&limit=3")])
      .then(([overview, catalog]) => { setDashboard(overview); setModels((catalog.items || catalog).filter((item) => item.installed).slice(0, 3)); })
      .catch((requestError) => setError(requestError.message));
    return () => { void request; };
  }, []);

  const stats = [
    ["Générations", dashboard?.generations_total ?? "—", BsLightning],
    ["Vidéos créées", dashboard?.videos_created ?? "—", BsCameraVideo],
    ["Images créées", dashboard?.images_created ?? "—", BsImage],
    ["Stockage utilisé", dashboard ? `${(dashboard.storage_bytes / 1073741824).toFixed(2)} Go` : "—", BsDatabase],
  ];

  return <div className={styles.page}>
    <header className={styles.pageHeading}><div><h1>Bonjour, Alex ! 👋</h1><p>Prêt à créer quelque chose d’incroyable aujourd’hui ?</p></div><Link className={styles.primaryButton} href="/generations"><BsPlus /> Nouveau projet</Link></header>
    {error && <div className={styles.errorBanner}>{error}</div>}
    <section className={styles.dashboardStats}>{stats.map(([label, value, Icon]) => <article key={label}><div><span>{label}</span><strong>{value}</strong></div><Icon /></article>)}</section>
    <div className={styles.dashboardSectionTitle}><h2>Modèles installés</h2><Link href="/models">Voir tous les modèles <BsArrowRight /></Link></div>
    <section className={styles.dashboardModels}>{models.map((model) => <Link href={`/models/${encodeURIComponent(model.id)}`} key={model.id}><div><BsStars /></div><h3>{model.name}</h3><span>{model.engine}</span><small>Prêt</small></Link>)}</section>
    <div className={styles.dashboardSectionTitle}><h2>Générations récentes</h2><Link href="/generations">Voir tout <BsArrowRight /></Link></div>
    <section className={styles.dashboardRecent}>{dashboard?.recent_generations?.map((item) => <article key={item.id}><div><strong>{item.prompt}</strong><span>{item.mode.replace("_TO_", " → ").replaceAll("_", " ")} · {item.model_id}</span></div><em>{item.status}</em><small>{item.progress}%</small></article>)}{dashboard && !dashboard.recent_generations.length && <p>Aucune génération récente.</p>}</section>
  </div>;
}
