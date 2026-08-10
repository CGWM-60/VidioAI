"use client";

import { useEffect, useState } from "react";
import { BsArrowCounterclockwise, BsCheck2, BsDatabase, BsFolder2Open, BsGear, BsHddStack } from "react-icons/bs";
import { apiFetch } from "../lib/api";
import styles from "../studio.module.css";

const PATH_FIELDS = [
  ["models_dir", "Dossier des modèles", "Poids et manifestes installés"],
  ["outputs_dir", "Sorties", "Assets et générations terminées"],
  ["cache_dir", "Cache", "Téléchargements réutilisables"],
  ["work_dir", "Dossier temporaire", "Fichiers de travail atomiques"],
];

export default function SettingsPage() {
  const [settings, setSettings] = useState(null);
  const [initial, setInitial] = useState(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    apiFetch("/api/settings").then((data) => {
      setSettings(data); setInitial(data);
    }).catch((requestError) => setError(requestError.message));
  }, []);

  function update(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
    setMessage("");
  }

  async function save(event) {
    event.preventDefault();
    setSaving(true); setError(""); setMessage("");
    try {
      const saved = await apiFetch("/api/settings", { method: "PUT", body: JSON.stringify(settings) });
      setSettings(saved); setInitial(saved);
      setMessage("Paramètres enregistrés et dossiers vérifiés.");
    } catch (requestError) { setError(requestError.message); }
    finally { setSaving(false); }
  }

  if (!settings) return <div className={styles.page}><div className={styles.stateCard}>{error || "Chargement de la configuration…"}</div></div>;

  return (
    <form className={styles.page} onSubmit={save}>
      <header className={styles.pageHeading}><div><h1>Paramètres</h1><p>Personnalisez le stockage et le comportement local de VidioAI.</p></div></header>
      {error && <div className={styles.errorBanner} role="alert">{error}</div>}
      {message && <div className={styles.successBanner}><BsCheck2 /> {message}</div>}

      <div className={styles.settingsGrid}>
        <section className={styles.settingsPanel}>
          <h2><BsDatabase /> Stockage</h2>
          {PATH_FIELDS.map(([key, label, help]) => (
            <label className={styles.pathField} key={key}>
              <span><strong>{label}</strong><small>{help}</small></span>
              <span className={styles.pathInput}><input value={settings[key]} onChange={(event) => update(key, event.target.value)} /><BsFolder2Open /></span>
            </label>
          ))}
        </section>

        <section className={styles.settingsPanel}>
          <h2><BsGear /> Gestion des modèles</h2>
          <label className={styles.settingRow}>
            <span><strong>Durée avant déchargement</strong><small>Libère un modèle après une période d’inactivité.</small></span>
            <select value={settings.auto_unload_minutes} onChange={(event) => update("auto_unload_minutes", Number(event.target.value))}>
              <option value={5}>5 minutes</option><option value={15}>15 minutes</option><option value={30}>30 minutes</option><option value={60}>1 heure</option>
            </select>
          </label>
          <label className={styles.settingRow}>
            <span><strong>Optimisation automatique</strong><small>Choisit la variante adaptée à la RAM et à la VRAM.</small></span>
            <button type="button" role="switch" aria-checked={settings.automatic_optimization} className={`${styles.toggle} ${settings.automatic_optimization ? styles.toggleOn : ""}`} onClick={() => update("automatic_optimization", !settings.automatic_optimization)}><span /></button>
          </label>
          <div className={styles.settingsNote}><BsHddStack /><p>Les chemins invalides sont refusés par Rust. Les dossiers valides manquants sont créés à la sauvegarde.</p></div>
        </section>
      </div>

      <footer className={styles.settingsFooter}>
        <button type="button" className={styles.secondaryButton} disabled={saving || JSON.stringify(initial) === JSON.stringify(settings)} onClick={() => setSettings(initial)}><BsArrowCounterclockwise /> Annuler les modifications</button>
        <button type="submit" className={styles.primaryButton} disabled={saving}><BsCheck2 /> {saving ? "Enregistrement…" : "Enregistrer les paramètres"}</button>
      </footer>
    </form>
  );
}
