"use client";

import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BsArrowClockwise, BsCameraVideo, BsCheckCircle, BsClock, BsDownload,
  BsFilm, BsImage, BsInfoCircle, BsPlayCircle, BsStars, BsUpload, BsXCircle,
} from "react-icons/bs";
import { FaArrowsAlt } from "react-icons/fa";
import { apiFetch, assetUrl, closeWebSocketSafely, eventsUrl } from "../lib/api";
import { generationFromJob } from "../lib/generation-job-state.mjs";
import styles from "../studio.module.css";

const MODES = [
  { id: "TEXT_TO_VIDEO", label: "Texte → Vidéo", icon: BsStars, capability: "TEXT_TO_VIDEO", mode: "TEXT_TO_VIDEO", inputKind: "none", accepts: "" },
  { id: "IMAGE_TO_VIDEO", label: "Image → Vidéo", icon: BsImage, capability: "IMAGE_TO_VIDEO", mode: "IMAGE_TO_VIDEO", inputKind: "image", accepts: "image/png,image/jpeg,image/webp" },
  { id: "MULTI_IMAGE_TO_VIDEO", label: "Multi-images → Vidéo", icon: BsImage, capability: "MULTI_IMAGE_TO_VIDEO", mode: "IMAGE_TO_VIDEO", inputKind: "image", accepts: "image/png,image/jpeg,image/webp" },
  { id: "START_END_IMAGE_TO_VIDEO", label: "Start/End → Vidéo", icon: FaArrowsAlt, capability: "START_END_IMAGE_TO_VIDEO", mode: "IMAGE_TO_VIDEO", inputKind: "image", accepts: "image/png,image/jpeg,image/webp" },
  { id: "KEYFRAMES_TO_VIDEO", label: "Keyframes → Vidéo", icon: BsImage, capability: "KEYFRAMES_TO_VIDEO", mode: "IMAGE_TO_VIDEO", inputKind: "image", accepts: "image/png,image/jpeg,image/webp" },
  { id: "VIDEO_TO_VIDEO", label: "Vidéo → Vidéo", icon: BsFilm, capability: "VIDEO_TO_VIDEO", mode: "VIDEO_TO_VIDEO", inputKind: "video", accepts: "video/mp4" },
  { id: "VIDEO_INPAINTING", label: "Inpainting vidéo", icon: BsFilm, capability: "VIDEO_INPAINTING", mode: "VIDEO_TO_VIDEO", inputKind: "video", accepts: "video/mp4" },
  { id: "VIDEO_UPSCALE", label: "Upscale vidéo", icon: BsFilm, capability: "VIDEO_UPSCALE", mode: "VIDEO_TO_VIDEO", inputKind: "video", accepts: "video/mp4" },
];
const DEFAULT_INPUT_PROFILE = {
  min_input_images: 1,
  max_input_images: 1,
  supported_image_roles: [],
  supports_start_end_frames: false,
  supports_reference_images: false,
  supports_keyframes: false,
};

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

function GenerationsContent() {
  const sourceAssetId = useSearchParams().get("asset");
  const fileInputRef = useRef(null);
  const [mode, setMode] = useState(sourceAssetId ? "IMAGE_TO_VIDEO" : "TEXT_TO_VIDEO");
  const [inputAsset, setInputAsset] = useState(sourceAssetId ? { id: sourceAssetId, kind: "IMAGE" } : null);
  const [inputImages, setInputImages] = useState(sourceAssetId ? [{ asset_id: sourceAssetId, order: 0, role: "start_frame" }] : []);
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState("vidio-motion-local");
  const [prompt, setPrompt] = useState("Une voiture volante traverse une ville futuriste au coucher du soleil, mouvement de caméra cinématographique.");
  const [duration, setDuration] = useState(6);
  const [quality, setQuality] = useState("480p");
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [fps, setFps] = useState(24);
  const [audio, setAudio] = useState(false);
  const [generation, setGeneration] = useState(null);
  const [activeJobId, setActiveJobId] = useState(() => typeof window === "undefined" ? "" : window.localStorage.getItem("vidioai.videos.activeJob") || "");
  const [history, setHistory] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const activeMode = MODES.find((item) => item.id === mode) || MODES[0];
  const generationId = generation?.id;
  const generationStatus = generation?.status;
  const compatibleModels = useMemo(() => models.filter((model) => (
    model.runtime_capabilities?.includes(activeMode.capability)
  )), [activeMode.capability, models]);
  const effectiveInputProfile = useMemo(() => {
    if (activeMode.inputKind !== "image") {
      return DEFAULT_INPUT_PROFILE;
    }
    const selectedModel = models.find((model) => model.id === modelId);
    return selectedModel?.input_profile || DEFAULT_INPUT_PROFILE;
  }, [activeMode.inputKind, modelId, models]);
  const visibleInputImages = useMemo(() => inputImages.slice(0, effectiveInputProfile.max_input_images), [effectiveInputProfile.max_input_images, inputImages]);

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
      if (envelope.event === "job.updated" && envelope.data?.id === activeJobId) {
        setGeneration((current) => generationFromJob(current, envelope.data));
        if (TERMINAL_STATUSES.has(envelope.data.status)) void refreshHistory();
      } else if (envelope.data?.id === generationId && envelope.event.startsWith("generation.")) {
        setGeneration(envelope.data);
        if (TERMINAL_STATUSES.has(envelope.data.status)) void refreshHistory();
      }
    };
    const interval = window.setInterval(async () => {
      try {
        const job = activeJobId ? await apiFetch(`/api/jobs/${activeJobId}`) : null;
        const reconciled = job?.result?.generation || await apiFetch(`/api/generations/${generationId}`);
        const current = job ? generationFromJob(reconciled, job) : reconciled;
        setGeneration(current);
        if (job && TERMINAL_STATUSES.has(job.status)) {
          window.localStorage.removeItem("vidioai.videos.activeJob");
          setActiveJobId("");
        }
        if (TERMINAL_STATUSES.has(current.status)) {
          window.clearInterval(interval);
          void refreshHistory();
        }
      } catch (requestError) {
        setError(requestError.message);
      }
    }, 700);
    return () => { closeWebSocketSafely(socket); window.clearInterval(interval); };
  }, [activeJobId, generationId, generationStatus, refreshHistory]);

  useEffect(() => {
    if (!activeJobId || generationId) return;
    let stopped = false;
    apiFetch(`/api/jobs/${activeJobId}`)
      .then((job) => job.result?.generation
        ? job.result.generation
        : apiFetch(`/api/generations/${job.target_id}`))
      .then((current) => { if (!stopped) setGeneration(current); })
      .catch((requestError) => { if (!stopped) setError(requestError.message); });
    return () => { stopped = true; };
  }, [activeJobId, generationId]);

  function chooseMode(nextMode) {
    setMode(nextMode);
    setInputAsset(null);
    setInputImages([]);
    setGeneration(null);
    setError("");
    const requiredCapability = MODES.find((item) => item.id === nextMode).capability;
    const local = models.find((model) => model.id === "vidio-motion-local" && model.runtime_capabilities?.includes(requiredCapability));
    setModelId(local?.id || models.find((model) => model.runtime_capabilities?.includes(requiredCapability))?.id || "");
  }

  async function uploadFile(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    setUploading(true);
    setError("");
    try {
      const assets = [];
      for (const file of files) {
        const body = new FormData();
        body.append("file", file);
        assets.push(await apiFetch("/api/assets", { method: "POST", body }));
      }
      if (activeMode.inputKind === "image") {
        const nextImages = [...inputImages];
        for (const asset of assets) {
          if (nextImages.length >= effectiveInputProfile.max_input_images) break;
          nextImages.push({ asset_id: asset.id, order: nextImages.length, role: nextImages.length === 0 ? "start_frame" : "reference" });
        }
        setInputImages(nextImages);
        setInputAsset(assets[0] || null);
      } else {
        setInputAsset(assets[0] || null);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function submitGeneration() {
    setError("");
    if (activeMode.inputKind !== "none" && !inputAsset && !visibleInputImages.length) {
      setError("Ajoutez le média de départ avant de lancer la génération.");
      return;
    }
    try {
      const payload = {
        mode: activeMode.mode,
        capability: activeMode.capability,
        prompt,
        negative_prompt: null,
        model_id: modelId,
        input_asset_id: activeMode.inputKind === "image" && visibleInputImages.length ? visibleInputImages[0].asset_id : inputAsset?.id,
        input_images: activeMode.inputKind === "image" ? visibleInputImages : [],
        duration_seconds: Number(duration),
        quality,
        aspect_ratio: aspectRatio,
        fps: Number(fps),
        audio,
      };
      const created = await apiFetch("/api/videos/generate", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setGeneration(created);
      setActiveJobId(created.job_id);
      window.localStorage.setItem("vidioai.videos.activeJob", created.job_id);
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

  function removeImage(index) {
    setInputImages((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }

  function moveImage(index, direction) {
    setInputImages((current) => {
      const next = [...current];
      const target = index + direction;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return next.map((item, itemIndex) => ({ ...item, order: itemIndex }));
    });
  }

  function updateImageRole(index, role) {
    setInputImages((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, role } : item));
  }

  const sourceIsVideo = inputAsset?.kind === "VIDEO";
  const outputId = generation?.output_asset_id;
  const isRunning = generation && !TERMINAL_STATUSES.has(generation.status);
  const selectedModel = models.find((model) => model.id === modelId);

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

          {activeMode.inputKind !== "none" && (
            <div className={styles.videoSourceBlock}>
              <div className={styles.sectionTitle}><strong>{sourceIsVideo ? "Vidéo de départ" : activeMode.inputKind === "image" ? "Images de départ" : "Image de départ"}</strong><span>{activeMode.inputKind === "image" ? `${visibleInputImages.length}/${effectiveInputProfile.max_input_images}` : "Asset persistant"}</span></div>
              {activeMode.inputKind === "image" ? (
                <div className={styles.imageStack}>
                  {visibleInputImages.length ? visibleInputImages.map((item, index) => (
                    <div key={`${item.asset_id}-${index}`} className={styles.imageStackCard}>
                      <div className={styles.imageStackNumber}>#{index + 1}</div>
                      <Image unoptimized width={240} height={160} src={assetUrl(item.asset_id)} alt={`Image ${index + 1}`} />
                      <div className={styles.imageStackActions}>
                        <button type="button" onClick={() => moveImage(index, -1)} disabled={index === 0} aria-label="Déplacer vers le haut">↑</button>
                        <button type="button" onClick={() => moveImage(index, 1)} disabled={index === visibleInputImages.length - 1} aria-label="Déplacer vers le bas">↓</button>
                        <button type="button" onClick={() => removeImage(index)} aria-label="Supprimer l'image"><BsXCircle /></button>
                      </div>
                      <label className={styles.formGroup}>
                        <span>Rôle</span>
                        <select value={item.role} onChange={(event) => updateImageRole(index, event.target.value)}>
                          <option value="start_frame">Start frame</option>
                          <option value="end_frame">End frame</option>
                          <option value="reference">Référence</option>
                          <option value="keyframe">Keyframe</option>
                        </select>
                      </label>
                    </div>
                  )) : null}
                  {visibleInputImages.length < effectiveInputProfile.max_input_images && (
                    <button type="button" className={styles.videoUpload} onClick={() => fileInputRef.current?.click()}>
                      <BsUpload /><strong>{uploading ? "Import en cours…" : "Ajouter une image"}</strong>
                      <span>{effectiveInputProfile.max_input_images === 1 ? "1 image maximum" : `Jusqu'à ${effectiveInputProfile.max_input_images} images`}</span>
                    </button>
                  )}
                  <input ref={fileInputRef} hidden type="file" accept={activeMode.accepts} multiple={effectiveInputProfile.max_input_images > 1} onChange={uploadFile} />
                </div>
              ) : (
                <>
                  {inputAsset ? (
                    <div className={styles.videoSourcePreview}>
                      {sourceIsVideo ? <video src={assetUrl(inputAsset.id)} controls preload="metadata" /> : <Image unoptimized width={960} height={540} src={assetUrl(inputAsset.id)} alt="Média de départ" />}
                      <button type="button" onClick={() => setInputAsset(null)} aria-label="Retirer le média"><BsXCircle /></button>
                    </div>
                  ) : (
                    <button type="button" className={styles.videoUpload} onClick={() => fileInputRef.current?.click()}>
                      <BsUpload /><strong>{uploading ? "Import en cours…" : "Glissez-déposez ou cliquez pour parcourir"}</strong>
                      <span>{activeMode.inputKind === "image" ? "PNG, JPEG ou WebP · 25 Mo max" : "MP4 · 512 Mo max"}</span>
                    </button>
                  )}
                  <input ref={fileInputRef} hidden type="file" accept={activeMode.accepts} onChange={uploadFile} />
                </>
              )}
            </div>
          )}

          <label className={styles.formGroup}><span>Prompt <small>{prompt.length} / 1000</small></span><textarea rows={5} maxLength={1000} value={prompt} onChange={(event) => setPrompt(event.target.value)} /></label>
          <label className={styles.formGroup}><span>Modèle</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}>
            {compatibleModels.map((model) => <option key={model.id} value={model.id}>{model.name}{model.runtime_ready ? " · prêt" : model.installed ? " · installé" : " · à installer"}</option>)}
          </select></label>
          {selectedModel?.installed && !selectedModel.runtime_ready && <div className={styles.warningBanner}>Ce modèle est installé mais pas chargé. Ouvrez sa fiche et lancez « Charger le modèle » avant la génération.</div>}

          <div className={styles.videoSettingsRow}>
            <label className={styles.formGroup}><span>Durée</span><select value={duration} onChange={(event) => setDuration(event.target.value)}><option value="4">4 secondes</option><option value="6">6 secondes</option><option value="10">10 secondes</option><option value="15">15 secondes</option></select></label>
            <label className={styles.formGroup}><span>Qualité</span><select value={quality} onChange={(event) => setQuality(event.target.value)}><option value="480p">Rapide 480p</option><option value="720p">HD 720p</option><option value="1080p">Full HD 1080p</option></select></label>
            <label className={styles.formGroup}><span>Ratio</span><select value={aspectRatio} onChange={(event) => setAspectRatio(event.target.value)}><option value="16:9">16:9 paysage</option><option value="9:16">9:16 portrait</option><option value="1:1">1:1 carré</option></select></label>
            <label className={styles.formGroup}><span>Cadence</span><select value={fps} onChange={(event) => setFps(event.target.value)}><option value="12">12 fps</option><option value="24">24 fps</option><option value="30">30 fps</option></select></label>
          </div>
          <div className={styles.audioSetting}><span><strong>Ajouter une piste audio silencieuse</strong><small>Crée une piste AAC vide prête pour un mixage ultérieur.</small></span><button type="button" role="switch" aria-checked={audio} onClick={() => setAudio((value) => !value)} className={`${styles.toggle} ${audio ? styles.toggleOn : ""}`}><span /></button></div>
          <button type="button" className={styles.generateButton} disabled={isRunning || !modelId || !selectedModel?.runtime_ready || prompt.trim().length < 3} onClick={submitGeneration}><BsStars /> {isRunning ? "Génération en cours…" : "Générer la vidéo"}</button>
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
            <div><BsPlayCircle /><span>{item.duration_seconds || "—"}s</span></div><strong>{item.prompt}</strong><small>{item.status} · {item.requested_quality || item.resolution || "—"} · {item.requested_aspect_ratio || "—"}{item.actual_width && item.actual_height ? ` · ${item.actual_width}×${item.actual_height}` : ""}</small>
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
