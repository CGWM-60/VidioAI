"use client";

import Link from "next/link";
import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  BsArrowRight, BsCheckCircleFill, BsCloudUpload, BsDownload,
  BsImage, BsInfoCircle, BsStars,
} from "react-icons/bs";
import { apiFetch, assetUrl, closeWebSocketSafely, eventsUrl } from "../lib/api";
import { generationFromJob, TERMINAL_JOB_STATUSES } from "../lib/generation-job-state.mjs";
import styles from "../studio.module.css";

// Les capacités sont des identifiants d'API, pas des phrases. Ce mapping évite
// les transformations naïves comme `TEXT → TO → IMAGE` et garde les messages
// français lisibles sans modifier la valeur envoyée au backend.
const CAPABILITY_LABELS = {
  TEXT_TO_IMAGE: "Texte → Image",
  IMAGE_TO_IMAGE: "Image → Image",
  INPAINTING: "Inpainting",
  OUTPAINTING: "Outpainting",
  IMAGE_VARIATION: "Variation d’image",
  IMAGE_UPSCALE: "Upscale image",
  CONTROLLED_IMAGE_GENERATION: "Génération contrôlée",
};

const IMAGE_MODES = [
  { id: "TEXT_TO_IMAGE", label: "Texte → Image", needsSource: false },
  { id: "IMAGE_TO_IMAGE", label: "Image → Image", needsSource: true },
  { id: "INPAINTING", label: "Inpainting", needsSource: true },
  { id: "OUTPAINTING", label: "Outpainting", needsSource: true },
  { id: "IMAGE_VARIATION", label: "Variation", needsSource: true },
  { id: "IMAGE_UPSCALE", label: "Upscale", needsSource: true },
  { id: "CONTROLLED_IMAGE_GENERATION", label: "Contrôle", needsSource: true },
];

export default function ImagesPage() {
  const fileInput = useRef(null);
  const maskInput = useRef(null);
  const controlInput = useRef(null);
  const [mode, setMode] = useState("TEXT_TO_IMAGE");
  const [models, setModels] = useState([]);
  // UNKNOWN est volontairement conservateur : tant que le backend n'a pas
  // confirmé le profil LOCAL, aucun moteur procédural ne peut être proposé.
  const [profile, setProfile] = useState("UNKNOWN");
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("Une voiture volante traverse une ville futuriste au coucher du soleil, ambiance cinématographique.");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [imageQuality, setImageQuality] = useState("quality");
  const [sourceAsset, setSourceAsset] = useState(null);
  const [sourcePreview, setSourcePreview] = useState("");
  const [generation, setGeneration] = useState(null);
  const [activeJobId, setActiveJobId] = useState(() => typeof window === "undefined" ? "" : window.localStorage.getItem("vidioai.images.activeJob") || "");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [maskAsset, setMaskAsset] = useState(null);
  const [controlAsset, setControlAsset] = useState(null);
  const [error, setError] = useState("");
  const activeMode = IMAGE_MODES.find((item) => item.id === mode) || IMAGE_MODES[0];

  // Seuls les modèles image réellement installés et compatibles apparaissent.
  //
  // Le catalogue et la readiness sont volontairement chargés séparément : un
  // Host Agent momentanément indisponible ne doit pas effacer les modèles déjà
  // connus. En revanche, une panne du catalogue empêche réellement de choisir
  // un moteur et reste donc présentée à l'utilisateur.
  useEffect(() => {
    apiFetch("/api/models?category=IMAGE&installed=true&compatible=true&limit=60&sort=compatibility")
      .then((catalog) => {
        const available = (catalog.items || catalog).filter((model) => model.kind === "IMAGE" && model.installed && model.compatible);
        setModels(available);
      })
      .catch((requestError) => setError(requestError.message));

    apiFetch("/api/ready")
      .then((readiness) => setProfile(readiness.profile || "UNKNOWN"))
      // L'état UNKNOWN est plus sûr que LOCAL : si la readiness de production
      // tombe, l'interface ne réactive pas accidentellement un moteur factice.
      .catch(() => setProfile("UNKNOWN"));
  }, []);

  // En production GPU, les moteurs procéduraux ne sont jamais proposés comme
  // substitut silencieux. La capacité doit être déclarée par la matrice runtime.
  const availableModels = useMemo(() => models.filter((model) =>
    model.runtime_capabilities?.includes(mode)
      && (profile === "LOCAL" || model.engine_type === "ai")
  ), [mode, models, profile]);
  const selectedModelId = availableModels.some((model) => model.id === modelId)
    ? modelId
    : availableModels[0]?.id || "";

  // Un seul WebSocket reçoit la progression de toutes les générations. L'UUID
  // permet d'ignorer proprement les événements appartenant à une autre page.
  useEffect(() => {
    const socket = new WebSocket(eventsUrl());
    socket.onmessage = (message) => {
      const envelope = JSON.parse(message.data);
      if (envelope.event === "job.updated" && envelope.data.id === activeJobId) {
        setGeneration((current) => generationFromJob(current, envelope.data));
        if (TERMINAL_JOB_STATUSES.has(envelope.data.status)) setBusy(false);
      } else if (envelope.event.startsWith("generation.") && envelope.data.id === generation?.id) {
        setGeneration(envelope.data);
        if (["completed", "failed"].includes(envelope.data.status)) setBusy(false);
      }
    };
    return () => closeWebSocketSafely(socket);
  }, [activeJobId, generation?.id]);

  useEffect(() => {
    if (!activeJobId) return undefined;
    const reconcile = async () => {
      try {
        const job = await apiFetch(`/api/jobs/${activeJobId}`);
        if (job.result?.generation) setGeneration((current) => generationFromJob(current, job));
        else if (job.target_id) {
          const current = await apiFetch(`/api/generations/${job.target_id}`);
          setGeneration({ ...current, progress: job.progress });
        }
        if (TERMINAL_JOB_STATUSES.has(job.status)) {
          setBusy(false);
          setActiveJobId("");
          window.localStorage.removeItem("vidioai.images.activeJob");
        }
      } catch (requestError) { setError(requestError.message); }
    };
    void reconcile();
    const interval = window.setInterval(() => void reconcile(), 1500);
    return () => window.clearInterval(interval);
  }, [activeJobId]);

  const outputUrl = useMemo(() => assetUrl(generation?.output_asset_id), [generation]);

  async function upload(file) {
    if (!file) return;
    setUploading(true); setError("");
    const preview = URL.createObjectURL(file);
    setSourcePreview((previous) => { if (previous) URL.revokeObjectURL(previous); return preview; });
    try {
      const form = new FormData(); form.append("file", file);
      setSourceAsset(await apiFetch("/api/assets", { method: "POST", body: form }));
    } catch (requestError) { setError(requestError.message); setSourceAsset(null); }
    finally { setUploading(false); }
  }

  async function uploadExtra(file, setter) {
    if (!file) return;
    setUploading(true); setError("");
    try {
      const form = new FormData(); form.append("file", file);
      setter(await apiFetch("/api/assets", { method: "POST", body: form }));
    } catch (requestError) { setError(requestError.message); setter(null); }
    finally { setUploading(false); }
  }

  async function submit(event) {
    event.preventDefault();
    if (activeMode.needsSource && !sourceAsset) { setError("Ajoutez d’abord une image source valide."); return; }
    setBusy(true); setError(""); setGeneration(null);
    try {
      const created = await apiFetch("/api/images/generate", {
        method: "POST",
        body: JSON.stringify({
          mode: activeMode.needsSource ? "IMAGE_TO_IMAGE" : "TEXT_TO_IMAGE",
          capability: mode,
          prompt,
          negative_prompt: negativePrompt || null,
          model_id: selectedModelId,
          quality: imageQuality,
          input_asset_id: activeMode.needsSource ? sourceAsset.id : null,
          mask_asset_id: ["INPAINTING", "OUTPAINTING"].includes(mode) ? maskAsset?.id || null : null,
          control_asset_id: mode === "CONTROLLED_IMAGE_GENERATION" ? controlAsset?.id || null : null,
        }),
      });
      setGeneration(created);
      setActiveJobId(created.job_id);
      window.localStorage.setItem("vidioai.images.activeJob", created.job_id);
    } catch (requestError) { setError(requestError.message); setBusy(false); }
  }

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div><h1><BsStars /> Génération d’image</h1><p>Créez des images à partir d’un texte ou transformez une image existante.</p></div>
        <div className={styles.creditBadge}><span>Runtime</span><strong>{profile === "GPU_PRODUCTION" ? "CUDA Worker privé" : profile === "LOCAL" ? "Local & privé" : "Vérification en cours"}</strong></div>
      </header>
      {error && <div className={styles.errorBanner} role="alert">{error}</div>}

      <div className={styles.imageStudio}>
        <form className={styles.generatorPanel} onSubmit={submit}>
          <div className={styles.modeTabs}>
            {IMAGE_MODES.map((item) => (
              <button key={item.id} type="button" className={mode === item.id ? styles.activeTab : ""} onClick={() => setMode(item.id)}>
                {item.id === "TEXT_TO_IMAGE" ? <span>T</span> : <BsImage />} {item.label}
              </button>
            ))}
          </div>

          {activeMode.needsSource && (
            <div className={styles.formGroup}>
              <label>Image de référence <small>(obligatoire)</small></label>
              <input ref={fileInput} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => upload(event.target.files?.[0])} />
              <button type="button" className={styles.uploadZone} onClick={() => fileInput.current?.click()}>
                {sourcePreview ? <Image unoptimized width={1024} height={1024} src={sourcePreview} alt="Aperçu de l’image source" /> : <><BsCloudUpload /><strong>Glissez-déposez une image ici</strong><span>ou cliquez pour parcourir</span></>}
                {uploading && <em>Validation et enregistrement…</em>}
                {sourceAsset && <em><BsCheckCircleFill /> Asset {sourceAsset.id.slice(0, 8)} enregistré</em>}
              </button>
            </div>
          )}

          {["INPAINTING", "OUTPAINTING"].includes(mode) && (
            <div className={styles.formGroup}>
              <label>Masque (optionnel)</label>
              <input ref={maskInput} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => uploadExtra(event.target.files?.[0], setMaskAsset)} />
              <button type="button" className={styles.uploadZone} onClick={() => maskInput.current?.click()}>
                <strong>{maskAsset ? "Masque prêt" : "Ajouter un masque"}</strong>
                <span>{maskAsset ? `Asset ${maskAsset.id.slice(0, 8)}` : "Zones à modifier"}</span>
              </button>
            </div>
          )}

          {mode === "CONTROLLED_IMAGE_GENERATION" && (
            <div className={styles.formGroup}>
              <label>Image de contrôle (optionnel)</label>
              <input ref={controlInput} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => uploadExtra(event.target.files?.[0], setControlAsset)} />
              <button type="button" className={styles.uploadZone} onClick={() => controlInput.current?.click()}>
                <strong>{controlAsset ? "Contrôle prêt" : "Ajouter une image de contrôle"}</strong>
                <span>{controlAsset ? `Asset ${controlAsset.id.slice(0, 8)}` : "Pose/croquis/contrainte"}</span>
              </button>
            </div>
          )}

          <label className={styles.formGroup}><span>Prompt <small>{prompt.length} / 1000</small></span><textarea maxLength={1000} rows={5} value={prompt} onChange={(event) => setPrompt(event.target.value)} /></label>
          <label className={styles.formGroup}><span>Prompt négatif <small>(optionnel)</small></span><textarea maxLength={1000} rows={2} value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} placeholder="flou, texte, watermark…" /></label>
          <label className={styles.formGroup}><span>Modèle</span><select value={selectedModelId} onChange={(event) => setModelId(event.target.value)}>{availableModels.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.engine}</option>)}</select></label>
          <label className={styles.formGroup}><span>Qualité</span><select value={imageQuality} onChange={(event) => setImageQuality(event.target.value)}><option value="quality">Quality</option><option value="balanced">Balanced</option><option value="fast">Fast</option><option value="native">Native pipeline</option></select></label>
          {!availableModels.length && <div className={styles.warningBanner}>Aucun modèle {CAPABILITY_LABELS[mode] || mode} installé et READY. Le pipeline n’est pas remplacé par une génération factice.</div>}

          <div className={styles.presetGrid}>
            <div className={styles.selectedPreset}><BsCheckCircleFill /><strong>Réaliste</strong><small>Style</small></div>
            <div><strong>1:1</strong><small>Ratio</small></div>
            <div><strong>{imageQuality === "quality" ? "Quality" : imageQuality}</strong><small>Recette</small></div>
          </div>
          <button className={styles.generateButton} disabled={busy || uploading || !selectedModelId || prompt.trim().length < 3}><BsStars /> {busy ? "Génération en cours…" : `Lancer ${CAPABILITY_LABELS[mode] || "la génération"}`}</button>
          <p className={styles.costNote}>Exécution locale · aucun crédit externe utilisé</p>
        </form>

        <section className={styles.resultPanel}>
          <div className={styles.resultMeta}>
            <div><span>Modèle</span><strong>{availableModels.find((model) => model.id === selectedModelId)?.name || "—"}</strong></div>
            <div><span>Progression</span><strong>{generation?.progress || 0} %</strong></div>
            <div><span>Statut</span><strong className={generation?.status === "completed" ? styles.statusReady : ""}>{generation?.status || "Prêt"}</strong></div>
          </div>

          <div className={styles.imageResult}>
            {outputUrl ? <Image unoptimized width={1024} height={1024} src={outputUrl} alt="Image générée à partir du prompt" /> : (
              <div className={styles.resultPlaceholder}><BsStars /><h2>{generation ? "Création de votre image…" : "Votre génération apparaîtra ici"}</h2><p>Le fichier final provient directement de l’asset stocké par Rust.</p></div>
            )}
            {generation && !outputUrl && <div className={styles.overlayProgress}><span style={{ width: `${generation.progress}%` }} /></div>}
          </div>

          {outputUrl && <div className={styles.resultActions}>
            <a className={styles.secondaryButton} href={outputUrl} download><BsDownload /> Télécharger</a>
            <Link className={styles.primaryButton} href={`/generations?asset=${generation.output_asset_id}`}><BsArrowRight /> Créer une vidéo</Link>
          </div>}
          <div className={styles.privacyNote}><BsInfoCircle /><span>Vos générations restent privées et sont servies par identifiant.</span></div>
        </section>
      </div>
    </div>
  );
}
