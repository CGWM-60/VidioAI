"use client";

import { useCallback, useEffect, useState } from "react";
import { BsFolder2Open, BsPlus, BsTrash } from "react-icons/bs";
import { apiFetch } from "../lib/api";
import styles from "../studio.module.css";

export default function ProjectsPage() {
  const [projects, setProjects] = useState([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const refresh = useCallback(() => apiFetch("/api/projects").then(setProjects).catch((requestError) => setError(requestError.message)), []);
  useEffect(() => { const request = Promise.resolve().then(refresh); return () => { void request; }; }, [refresh]);

  async function createProject(event) {
    event.preventDefault(); setError("");
    try {
      await apiFetch("/api/projects", { method: "POST", body: JSON.stringify({ name, description: "", asset_ids: [], generation_ids: [], chat_ids: [] }) });
      setName(""); await refresh();
    } catch (requestError) { setError(requestError.message); }
  }

  async function removeProject(id) {
    try { await apiFetch(`/api/projects/${id}`, { method: "DELETE" }); await refresh(); }
    catch (requestError) { setError(requestError.message); }
  }

  return <div className={styles.page}>
    <header className={styles.pageHeading}><div><h1>Projets</h1><p>Regroupez vos assets, conversations et générations.</p></div></header>
    {error && <div className={styles.errorBanner}>{error}</div>}
    <form className={styles.projectCreate} onSubmit={createProject}><input value={name} onChange={(event) => setName(event.target.value)} placeholder="Nom du nouveau projet" minLength={2} maxLength={80} required /><button className={styles.primaryButton}><BsPlus /> Créer</button></form>
    <section className={styles.projectGrid}>{projects.map((project) => <article key={project.id}><BsFolder2Open /><div><h2>{project.name}</h2><p>{project.description || "Projet créatif VidioAI"}</p><span>{project.asset_ids.length} assets · {project.generation_ids.length} générations · {project.chat_ids.length} chats</span></div><button type="button" onClick={() => void removeProject(project.id)} aria-label={`Supprimer ${project.name}`}><BsTrash /></button></article>)}{!projects.length && <div className={styles.stateCard}><BsFolder2Open /> Aucun projet pour le moment.</div>}</section>
  </div>;
}
