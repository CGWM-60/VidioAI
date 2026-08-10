"use client";

import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BsArrowClockwise, BsCameraVideo, BsCheckCircle, BsClock, BsDownload,
  BsFilm, BsImage, BsInfoCircle, BsPlayCircle, BsStars, BsUpload, BsXCircle,
} from "react-icons/bs";
import { apiFetch, assetUrl, closeWebSocketSafely, eventsUrl } from "../lib/api";
import styles from "../studio.module.css";

const MODES = [
  { id: "TEXT_TO_VIDEO", label: "Texte → Vidéo", icon: BsStars, capability: "TEXT_TO_VIDEO", accepts: "" },
  { id: "IMAGE_TO_VIDEO", label: "Image → Vidéo", icon: BsImage, capability: "IMAGE_TO_VIDEO", accepts: "image/png,image/jpeg,image/webp" },
  { id: "VIDEO_TO_VIDEO", label: "Vidéo → Vidéo", icon: BsFilm, capability: "VIDEO_TO_VIDEO", accepts: "video/mp4" },
];

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

function GenerationsContent() {
  const sourceAssetId = useSearchParams().get("asset");
  const fileInputRef = useRef(null);
  const [mode, setMode] = useState(sourceAssetId ? "IMAGE_TO_VIDEO" : "TEXT_TO_VIDEO");
  const [inputAsset, setInputAsset] = useState(sourceAssetId ? { id: sourceAssetId, kind: "IMAGE" } : null);
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState("vidio-motion-local");
  const [prompt, setPrompt] = useState("Une voiture volante traverse une ville futuriste au coucher du soleil, mouvement de caméra cinématographique.");
  const [duration, setDuration] = useState(6);
  const [resolution, setResolution] = useState("720p");
  const [audio, setAudio] = useState(false);
  const [generation, setGeneration] = useState(null);
  const [history, setHistory] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const activeMode = MODES.find((item) => item.id === mode);
  const generationId = generation?.id;
  const generationStatus = generation?.status;
  const compatibleModels = useMemo(() => models.filter((model) => (
    model.capabilities.includes(activeMode.capability)
  )), [activeMode.capability, models]);

  const refreshHistory = useCallback(async () => {
    const items = await apiFetch("/api/generations");
    setHistory(items.filter((item) => item.kind === "VIDEO").slice(0, 8));
  }, []);

  useEffect(() => {
    // Les appels sont différés dans une micro-tâche pour ne jamais modifier un
    // état pendant le cycle synchrone de montage React.
    const request = Promise.all([apiFetch("/api/models?category=VIDEO&installed=true&compatible=true&limit=60"), apiFetch("/api/generations")])
      .then(([catalog, generations]) => {
        setModels(catalog.items || catalog);
        setHistory(generations.filter((item) => item.kind === "VIDEO").slice(0, 8));
      })
      .catch((requestError) => setError(requestError.message));
    return () => { void request; };
  }, []);

  useEffect(() => {
    if (!generationId || TERMINAL_STATUSES.has(generationStatus)) return undefined;

    // Le WebSocket accélère l'interface ; le polling reste la source de secours
    // si un proxy ne relaie pas encore les upgrades WS en production.
    const socket = new WebSocket(eventsUrl());
    socket.onmessage = (event) => {
      const envelope = JSON.parse(event.data);
      if (envelope.data?.id !== generationId) return;
      if (envelope.event.startsWith("generation.")) {
        setGeneration(envelope.data);
        if (TERMINAL_STATUSES.has(envelope.data.status)) void refreshHistory();
      }
    };
    const interval = window.setInterval(async () => {
      try {
        const current = await apiFetch(`/api/generations/${generationId}`);
        setGeneration(current);
        if (TERMINAL_STATUSES.has(current.status)) {
          window.clearInterval(interval);
          void refreshHistory();
        }
      } catch (requestError) {
        setError(requestError.message);
      }
    }, 700);
    return () => { closeWebSocketSafely(socket); window.clearInterval(interval); };
  }, [generationId, generationStatus, refreshHistory]);

  function chooseMode(nextMode) {
    setMode(nextMode);
    setInputAsset(null);
    setGeneration(null);
    setError("");
    const requiredCapability = MODES.find((item) => item.id === nextMode).capability;
    const local = models.find((model) => model.id === "vidio-motion-local" && model.capabilities.includes(requiredCapability));
    setModelId(local?.id || models.find((model) => model.capabilities.includes(requiredCapability))?.id || "");
  }

  async function uploadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      setInputAsset(await apiFetch("/api/assets", { method: "POST", body }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function submitGeneration() {
    setError("");
    if (mode !== "TEXT_TO_VIDEO" && !inputAsset) {
      setError("Ajoutez le média de départ avant de lancer la génération.");
      return;
    }
    try {
      const created = await apiFetch("/api/videos/generate", {
        method: "POST",
        body: JSON.stringify({
          mode, prompt, model_id: modelId, input_asset_id: inputAsset?.id,
          duration_seconds: Number(duration), resolution, audio,
        }),
      });
      setGeneration(created);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function cancelGeneration() {
    try {
      await apiFetch(`/api/generations/${generation.id}/cancel`, { method: "POST" });
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  const sourceIsVideo = inputAsset?.kind === "VIDEO";
  const outputId = generation?.output_asset_id;
  const isRunning = generation && !TERMINAL_STATUSES.has(generation.status);

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div><h1><BsStars /> Génération vidéo</h1><p>Créez, animez ou transformez une vidéo en quelques secondes.</p></div>
        <button type="button" className={styles.secondaryButton} onClick={() => void refreshHistory()}><BsArrowClockwise /> Actualiser</button>
      </header>

      {error && <div className={styles.errorBanner} role="alert"><BsXCircle /> {error}</div>}

      <section className={styles.videoComposer}>
        <div className={styles.videoFormPanel}>
          <div className={styles.videoModeTabs}>
            {MODES.map((item) => {
              const Icon = item.icon;
              return <button type="button" key={item.id} onClick={() => chooseMode(item.id)} className={mode === item.id ? styles.videoModeActive : ""}><Icon /> {item.label}</button>;
            })}
          </div>

          {mode !== "TEXT_TO_VIDEO" && (
            <div className={styles.videoSourceBlock}>
              <div className={styles.sectionTitle}><strong>{sourceIsVideo ? "Vidéo de départ" : "Image de départ"}</strong><span>Asset persistant</span></div>
              {inputAsset ? (
                <div className={styles.videoSourcePreview}>
                  {sourceIsVideo ? <video src={assetUrl(inputAsset.id)} controls preload="metadata" /> : <Image unoptimized width={960} height={540} src={assetUrl(inputAsset.id)} alt="Média de départ" />}
                  <button type="button" onClick={() => setInputAsset(null)} aria-label="Retirer le média"><BsXCircle /></button>
                </div>
              ) : (
                <button type="button" className={styles.videoUpload} onClick={() => fileInputRef.current?.click()}>
                  <BsUpload /><strong>{uploading ? "Import en cours…" : "Glissez-déposez ou cliquez pour parcourir"}</strong>
                  <span>{mode === "IMAGE_TO_VIDEO" ? "PNG, JPEG ou WebP · 25 Mo max" : "MP4 · 512 Mo max"}</span>
                </button>
              )}
              <input ref={fileInputRef} hidden type="file" accept={activeMode.accepts} onChange={uploadFile} />
            </div>
          )}

          <label className={styles.formGroup}><span>Prompt <small>{prompt.length} / 1000</small></span><textarea rows={5} maxLength={1000} value={prompt} onChange={(event) => setPrompt(event.target.value)} /></label>
          <label className={styles.formGroup}><span>Modèle</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}>
            {compatibleModels.map((model) => <option key={model.id} value={model.id}>{model.name}{model.installed ? " · prêt" : " · à installer"}</option>)}
          </select></label>

          <div className={styles.videoSettingsRow}>
            <label className={styles.formGroup}><span>Durée</span><select value={duration} onChange={(event) => setDuration(event.target.value)}><option value="4">4 secondes</option><option value="6">6 secondes</option><option value="10">10 secondes</option><option value="15">15 secondes</option></select></label>
            <label className={styles.formGroup}><span>Qualité</span><select value={resolution} onChange={(event) => setResolution(event.target.value)}><option value="720p">HD 720p</option><option value="1080p">Full HD 1080p</option></select></label>
          </div>
          <div className={styles.audioSetting}><span><strong>Ajouter une piste audio silencieuse</strong><small>Crée une piste AAC vide prête pour un mixage ultérieur.</small></span><button type="button" role="switch" aria-checked={audio} onClick={() => setAudio((value) => !value)} className={`${styles.toggle} ${audio ? styles.toggleOn : ""}`}><span /></button></div>
          <button type="button" className={styles.generateButton} disabled={isRunning || !modelId || prompt.trim().length < 3} onClick={submitGeneration}><BsStars /> {isRunning ? "Génération en cours…" : "Générer la vidéo"}</button>
        </div>

        <div className={styles.videoResultPanel}>
          <div className={styles.resultTopline}><div><span>Modèle</span><strong>{models.find((model) => model.id === modelId)?.name || "—"}</strong></div><div><span>Temps estimé</span><strong><BsClock /> ~ {duration}s</strong></div><div><span>Statut</span><strong className={generation?.status === "completed" ? styles.statusReady : ""}>{generation?.status || "prêt"}</strong></div></div>
          <div className={styles.videoStage}>
            {outputId ? <video src={assetUrl(outputId)} controls autoPlay loop /> : inputAsset ? (sourceIsVideo ? <video src={assetUrl(inputAsset.id)} controls /> : <Image unoptimized width={1280} height={720} src={assetUrl(inputAsset.id)} alt="Aperçu de la génération" />) : <div className={styles.videoPlaceholder}><BsCameraVideo /><h2>Votre vidéo apparaîtra ici</h2><p>Choisissez un mode et décrivez le mouvement souhaité.</p></div>}
            {isRunning && <div className={styles.videoProgressOverlay}><strong>{generation.progress}%</strong><div><span style={{ width: `${generation.progress}%` }} /></div><p>Encodage et optimisation du résultat…</p></div>}
          </div>
          {generation?.status === "failed" && <div className={styles.errorBanner}>{generation.error}</div>}
          <div className={styles.videoResultActions}>
            {isRunning && <button type="button" className={styles.dangerButton} onClick={cancelGeneration}><BsXCircle /> Annuler</button>}
            {outputId && <a className={styles.primaryButton} href={assetUrl(outputId)} download><BsDownload /> Télécharger le MP4</a>}
          </div>
          <div className={styles.privacyNote}><BsInfoCircle /><span>Le média original est conservé ; chaque transformation crée un nouvel asset privé.</span></div>
        </div>
      </section>

      <section className={styles.videoHistory}>
        <div className={styles.sectionTitle}><div><strong>Résultats récents</strong><span>{history.length} génération(s)</span></div></div>
        <div className={styles.videoHistoryGrid}>
          {history.map((item) => <button type="button" key={item.id} onClick={() => setGeneration(item)} className={styles.historyCard}>
            <div><BsPlayCircle /><span>{item.duration_seconds || "—"}s</span></div><strong>{item.prompt}</strong><small>{item.status} · {item.resolution || "—"}</small>
          </button>)}
          {!history.length && <div className={styles.emptyHistory}><BsFilm /> Aucune vidéo générée pour le moment.</div>}
        </div>
      </section>
    </div>
  );
}

export default function GenerationsPage() {
  return <Suspense fallback={<div className={styles.page}><div className={styles.stateCard}>Préparation du studio vidéo…</div></div>}><GenerationsContent /></Suspense>;
}
