"use client";

import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { BsArrowLeft, BsCheck2, BsCloudDownload, BsCpu, BsPlay, BsStop } from "react-icons/bs";
import { apiFetch } from "../../lib/api";
import { MODEL_PREFLIGHT_TIMEOUT_MS } from "../catalog-state.mjs";
import styles from "../../studio.module.css";

function formatMemory(bytes) {
  if (!Number.isFinite(bytes)) return "Non disponible";
  const gib = bytes / 1073741824;
  return `${gib.toFixed(gib >= 10 ? 0 : 1)} Go`;
}

function formatRange(range) {
  if (!range || !Number.isFinite(range.min_bytes) || !Number.isFinite(range.max_bytes)) return "Non disponible";
  if (range.min_bytes === range.max_bytes) return formatMemory(range.min_bytes);
  return `~${formatMemory(range.min_bytes)}–${formatMemory(range.max_bytes)}`;
}

function sourceLabel(hardware) {
  if (hardware?.source === "measured") return `Mesuré sur ${hardware.benchmark?.gpu || "la machine actuelle"}`;
  if (hardware?.source === "official") return "Configuration officielle";
  if (hardware?.source === "estimated") return "Estimation matérielle";
  if (hardware?.source === "partial") return "Estimation partielle";
  return "Informations matérielles insuffisantes";
}

function runtimeStatus(model) {
  return model.runtime_compatibility || (model.runtime_supported ? "SUPPORTED" : "UNSUPPORTED");
}

function parseLoraLines(value) {
  return value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const [repository, scale = "1", weightName = "", revision = "main"] = line.split("|").map((item) => item.trim());
      if (!repository || !repository.includes("/")) throw new Error(`LoRA ligne ${index + 1}: owner/model attendu.`);
      const parsedScale = Number(scale);
      if (!Number.isFinite(parsedScale) || parsedScale < 0 || parsedScale > 2) throw new Error(`LoRA ligne ${index + 1}: scale doit etre entre 0 et 2.`);
      return {
        repository,
        revision: revision || "main",
        scale: parsedScale,
        enabled: true,
        ...(weightName ? { weight_name: weightName } : {}),
      };
    });
}

function optionalInteger(value) {
  if (`${value}`.trim() === "") return null;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) throw new Error(`Entier invalide: ${value}`);
  return parsed;
}

function optionalFloat(value) {
  if (`${value}`.trim() === "") return null;
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) throw new Error(`Nombre invalide: ${value}`);
  return parsed;
}

/** Page de détail alimentée par la route query-safe `/api/models/by-id`. */
function ModelDetailsContent() {
  const { id } = useParams();
  const queryModelId = useSearchParams().get("model_id");
  const router = useRouter();
  // La query string est canonique pour les repositories organisation/modèle.
  // Le segment dynamique reste toléré pour les anciens liens et moteurs locaux.
  const modelId = queryModelId || decodeURIComponent(id || "");
  const [model, setModel] = useState(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [error, setError] = useState("");
  const [loraText, setLoraText] = useState("");
  const [recipeQuality, setRecipeQuality] = useState("native");
  const [recipeWidth, setRecipeWidth] = useState("");
  const [recipeHeight, setRecipeHeight] = useState("");
  const [recipeSteps, setRecipeSteps] = useState("");
  const [recipeGuidance, setRecipeGuidance] = useState("");
  const [recipeTrueCfg, setRecipeTrueCfg] = useState("");
  const [recipeStrength, setRecipeStrength] = useState("");
  const [recipeMaxSequence, setRecipeMaxSequence] = useState("");
  const [recipeInferenceFps, setRecipeInferenceFps] = useState("");

  const refresh = useCallback(async () => {
    try {
      const nextModel = await apiFetch(`/api/models/by-id?model_id=${encodeURIComponent(modelId)}`);
      const bundle = nextModel.bundle || {};
      const recipe = bundle.recipe || {};

      setModel(nextModel);
      setLoraText((bundle.loras || []).map((item) => [
        item.repository,
        item.scale ?? 1,
        item.weight_name || "",
        item.requested_revision || item.revision || "main",
      ].join(" | ")).join("\n"));
      setRecipeQuality(recipe.quality_mode || "native");
      setRecipeWidth(recipe.width ?? "");
      setRecipeHeight(recipe.height ?? "");
      setRecipeSteps(recipe.num_inference_steps ?? "");
      setRecipeGuidance(recipe.guidance_scale ?? "");
      setRecipeTrueCfg(recipe.true_cfg_scale ?? "");
      setRecipeStrength(recipe.strength ?? "");
      setRecipeMaxSequence(recipe.max_sequence_length ?? "");
      setRecipeInferenceFps(recipe.inference_fps ?? "");
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    }
  }, [modelId]);

  useEffect(() => {
    const request = Promise.resolve().then(refresh);
    return () => { void request; };
  }, [refresh]);

  async function changeRuntime(action) {
    setRuntimeBusy(true);
    try {
      await apiFetch(`/api/models/${action}`, {
        method: "POST",
        body: JSON.stringify({ model_id: modelId }),
      });
      await refresh();
    }
    catch (requestError) { setError(requestError.message); }
    finally { setRuntimeBusy(false); }
  }

  async function startInstall() {
    setRuntimeBusy(true);
    try {
      const job = await apiFetch("/api/models/install", {
        method: "POST",
        timeoutMs: MODEL_PREFLIGHT_TIMEOUT_MS,
        timeoutCode: "MODEL_PREFLIGHT_TIMEOUT",
        body: JSON.stringify({
          model_id: model.id,
          revision: model.installed ? (model.installed_revision || model.revision) : model.revision,
          loras: parseLoraLines(loraText),
          recipe: {
            quality_mode: recipeQuality,
            width: optionalInteger(recipeWidth),
            height: optionalInteger(recipeHeight),
            num_inference_steps: optionalInteger(recipeSteps),
            guidance_scale: optionalFloat(recipeGuidance),
            true_cfg_scale: optionalFloat(recipeTrueCfg),
            strength: optionalFloat(recipeStrength),
            max_sequence_length: optionalInteger(recipeMaxSequence),
            inference_fps: optionalInteger(recipeInferenceFps),
          },
        }),
      });
      router.push(`/models/install?model_id=${encodeURIComponent(model.id)}&job=${job.id}`);
    } catch (requestError) { setError(requestError.message); setRuntimeBusy(false); }
  }

  async function retryCache() {
    setRuntimeBusy(true);
    try {
      const job = await apiFetch("/api/models/cache", {
        method: "POST",
        body: JSON.stringify({ model_id: model.id }),
      });
      router.push(`/models/install?model_id=${encodeURIComponent(model.id)}&job=${job.id}`);
    } catch (requestError) { setError(requestError.message); setRuntimeBusy(false); }
  }

  if (error && !model) return <div className={styles.page}><div className={styles.errorBanner}>{error}</div></div>;
  if (!model) return <div className={styles.page}><div className={styles.stateCard}>Chargement du modèle…</div></div>;

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div className={styles.headingWithBack}><Link href="/models"><BsArrowLeft /></Link><div><h1>{model.name}</h1><p>{model.description}</p></div></div>
      </header>
      {error && <div className={styles.errorBanner}>{error}</div>}
      <section className={`${styles.largePanel} ${styles.detailsGrid}`}>
        <div><span className={styles.eyebrow}>Moteur</span><h2>{model.engine}</h2></div>
        <div><span className={styles.eyebrow}>Licence</span><h2>{model.license}</h2></div>
        <div><span className={styles.eyebrow}>État local</span><h2>{model.loaded ? "Chargé" : model.installed ? "Installé" : "Absent"}</h2></div>
      </section>
      <section className={`${styles.largePanel} ${styles.detailsGrid}`}>
        <div><span className={styles.eyebrow}>Révision</span><h2>{model.revision.slice(0, 12)}</h2></div>
        <div><span className={styles.eyebrow}>Compatibilité</span><h2>{model.compatibility_level.replaceAll("_", " ")}</h2></div>
        <div><span className={styles.eyebrow}>Accès</span><h2>{model.accessibility.replaceAll("_", " ")}</h2></div>
      </section>
      {model.repository_url && <section className={styles.largePanel}><h2>Source officielle</h2><a className={styles.repositoryLink} href={model.repository_url} target="_blank" rel="noreferrer">{model.repository}</a></section>}
      <section className={styles.largePanel}>
        <h2>Bundle VidioAI</h2>
        <p><strong>Base :</strong> {model.bundle?.base_model?.repository || model.repository} @ {(model.bundle?.base_model?.revision || model.revision).slice(0, 12)}</p>
        <label className={styles.formGroup}>
          <span>LoRA(s) · une ligne par adapter</span>
          <textarea rows={4} value={loraText} onChange={(event) => setLoraText(event.target.value)} placeholder={"owner/lora | 0.8 | pytorch_lora_weights.safetensors | main\nowner/autre-lora | 0.5"} />
          <small>repository | scale | weight_name optionnel | revision optionnelle</small>
        </label>
        <div className={styles.detailsGrid}>
          <label className={styles.formGroup}><span>Profil</span><select value={recipeQuality} onChange={(event) => setRecipeQuality(event.target.value)}><option value="native">Native</option><option value="fast">Fast</option><option value="balanced">Balanced</option><option value="quality">Quality</option></select></label>
          <label className={styles.formGroup}><span>Largeur</span><input value={recipeWidth} onChange={(event) => setRecipeWidth(event.target.value)} placeholder="1024" /></label>
          <label className={styles.formGroup}><span>Hauteur</span><input value={recipeHeight} onChange={(event) => setRecipeHeight(event.target.value)} placeholder="1024" /></label>
          <label className={styles.formGroup}><span>Steps</span><input value={recipeSteps} onChange={(event) => setRecipeSteps(event.target.value)} placeholder="default natif si vide" /></label>
          <label className={styles.formGroup}><span>Guidance</span><input value={recipeGuidance} onChange={(event) => setRecipeGuidance(event.target.value)} placeholder="default natif si vide" /></label>
          <label className={styles.formGroup}><span>True CFG</span><input value={recipeTrueCfg} onChange={(event) => setRecipeTrueCfg(event.target.value)} placeholder="si supporte" /></label>
          <label className={styles.formGroup}><span>Strength I2I</span><input value={recipeStrength} onChange={(event) => setRecipeStrength(event.target.value)} placeholder="0 a 1" /></label>
          <label className={styles.formGroup}><span>Max sequence</span><input value={recipeMaxSequence} onChange={(event) => setRecipeMaxSequence(event.target.value)} placeholder="si supporte" /></label>
          <label className={styles.formGroup}><span>FPS inference</span><input value={recipeInferenceFps} onChange={(event) => setRecipeInferenceFps(event.target.value)} placeholder="distinct du FPS final" /></label>
        </div>
        <p className={styles.measuredNote}>Priorite : generation explicite → recette du bundle → default reel de la pipeline. Aucun branchement par nom de modele.</p>
      </section>
      <section className={styles.largePanel}>
        <h2>Capacités</h2>
        <div className={styles.capabilityCards}>{model.capabilities.map((item) => <span key={item}><BsCheck2 /> {item.replaceAll("_", " ")}</span>)}</div>
      </section>
      {model.runtime_dependencies?.length > 0 && <section className={styles.largePanel}>
        <h2>Dépendances runtime</h2>
        <div className={styles.dependencyList}>{model.runtime_dependencies.map((dependency) => (
          <div key={dependency.import_name}>
            <strong>✓ {dependency.package} {dependency.version}</strong>
            <span>{dependency.status} · {dependency.source} · requis par {dependency.required_by}</span>
          </div>
        ))}</div>
      </section>}
      {model.runtime_precision && <section className={styles.largePanel}>
        <h2>Précision runtime : {model.runtime_precision.precision}</h2>
        <p>{model.runtime_precision.source}{model.runtime_precision.quantization ? ` · ${model.runtime_precision.quantization}` : ""}</p>
        <div className={styles.dependencyList}>{model.runtime_precision.components?.map((component) => (
          <div key={component.name}>
            <strong>{component.name.replaceAll("_", " ")} · {component.dtype || "auto"}</strong>
            <span>{component.device || "device auto"}{component.quantization ? ` · ${component.quantization}` : ""}</span>
          </div>
        ))}</div>
      </section>}
      {["CACHE_FAILED", "CACHE_MANUAL"].includes(model.cache_status) && <section className={styles.largePanel}>
        <h2>{model.cache_status === "CACHE_MANUAL" ? "Sauvegarde S3 manuelle" : "Cache S3 à reprendre"}</h2>
        <p>{model.cache_status === "CACHE_MANUAL" ? "La sauvegarde automatique des modèles est temporairement désactivée. Le modèle est installé localement." : `Le modèle reste installé localement. ${model.cache_error || ""}`}</p>
        <button className={styles.secondaryButton} disabled={runtimeBusy} onClick={retryCache}>{model.cache_status === "CACHE_MANUAL" ? "Sauvegarder maintenant" : "Réessayer la sauvegarde S3"}</button>
      </section>}
      <section className={styles.largePanel}>
        <h2>Pourquoi ce modèle est-il utilisable ou non ?</h2>
        {runtimeStatus(model) === "UNKNOWN" && <div className={styles.warningBanner}>Compatibilité runtime inconnue avant téléchargement : le snapshot peut être installé puis chargé pour une validation locale réelle.</div>}
        <div className={styles.compatibilityChecks}>
          {model.compatibility_checks.map((check) => (
            <div className={check.ok ? styles.compatibilityOk : styles.compatibilityKo} key={check.key}>
              <strong>{check.ok ? "✓" : "✕"} {check.label}</strong>
              <span>{check.detail}</span>
            </div>
          ))}
        </div>
        <p className={styles.measuredNote}>Pipeline détecté : {model.pipeline_class || model.pipeline_tag || "non déterminé"}</p>
      </section>
      <section className={`${styles.largePanel} ${styles.hardwarePanel}`} title="Les estimations sont calculées à partir des poids et de la configuration Hugging Face. La consommation réelle varie selon la résolution et les paramètres de génération.">
        <div className={styles.hardwareHeading}>
          <div><span className={styles.eyebrow}>Configuration matérielle</span><h2>{sourceLabel(model.hardware)}</h2></div>
          <span className={styles.confidenceBadge}>Confiance {model.hardware.confidence === "high" ? "élevée" : model.hardware.confidence === "medium" ? "moyenne" : "faible"}</span>
        </div>
        {model.hardware.source !== "unknown" ? (
          <>
            <div className={styles.hardwareMetrics}>
              <div><span>VRAM d’inférence</span><strong>{model.hardware.source === "measured" ? formatMemory(model.hardware.benchmark?.vram_peak_bytes) : `~${formatMemory(model.hardware.estimated_vram_min)}–${formatMemory(model.hardware.estimated_vram_recommended)}`}</strong></div>
              <div><span>RAM</span><strong>{formatRange(model.hardware.estimated_ram)}</strong></div>
              <div><span>Poids seuls</span><strong>{formatRange(model.hardware.weights_memory)}</strong><small>Ce n’est pas la VRAM totale d’inférence.</small></div>
              <div><span>Backend</span><strong>{model.hardware.recommended_backend || "Non déterminé"}</strong></div>
              <div><span>Précision</span><strong>{model.hardware.recommended_precision || "Non déterminée"}</strong></div>
              <div><span>Compatibilité</span><strong>{model.hardware.compatibility_level.toLowerCase().replaceAll("_", " ")}</strong></div>
            </div>
            {model.hardware.benchmark?.inference_seconds && <p className={styles.measuredNote}>Temps mesuré : {model.hardware.benchmark.inference_seconds.toFixed(1)} s · batch {model.hardware.benchmark.batch}</p>}
            <ul className={styles.hardwareNotes}>{model.hardware.notes.map((note) => <li key={note}>{note}</li>)}</ul>
          </>
        ) : <p>Le repository ne fournit ni poids dimensionnés, ni métadonnées Safetensors, ni exigence officielle exploitable.</p>}
      </section>
      <section className={styles.largePanel}>
        <h2>Variantes</h2>
        <div className={styles.variantGrid}>{model.variants.map((variant) => (
          <div key={variant.id}><strong>{variant.label}</strong><span><BsCpu /> RAM {(variant.ram_required / 1073741824).toFixed(1)} Go · VRAM {(variant.vram_required / 1073741824).toFixed(1)} Go</span>{model.recommended_variant === variant.id && <em>Recommandée</em>}</div>
        ))}</div>
      </section>
      <div className={styles.footerActions}>
        {model.installed && !model.loaded && <button className={styles.primaryButton} disabled={runtimeBusy} onClick={() => changeRuntime("load")}><BsPlay /> Charger le modèle</button>}
        {model.loaded && <button className={styles.secondaryButton} disabled={runtimeBusy} onClick={() => changeRuntime("unload")}><BsStop /> Décharger</button>}
        {model.installed && <button className={styles.primaryButton} disabled={runtimeBusy} onClick={startInstall}><BsCloudDownload /> Appliquer LoRA / recette</button>}
        {!model.installed && <button className={styles.primaryButton} disabled={!model.installable || runtimeBusy} onClick={startInstall}><BsCloudDownload /> {model.gated && !model.access_authorized ? "Accès Hugging Face requis" : runtimeStatus(model) === "SUPPORTED" ? "Installer cette révision" : runtimeStatus(model) === "UNKNOWN" ? "Installer pour valider" : "Pipeline non implémenté"}</button>}
      </div>
    </div>
  );
}

export default function ModelDetailsPage() {
  return <Suspense fallback={<div className={styles.page}><div className={styles.stateCard}>Chargement du modèle…</div></div>}><ModelDetailsContent /></Suspense>;
}
