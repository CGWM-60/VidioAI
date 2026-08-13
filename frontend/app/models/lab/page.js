"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BsArrowClockwise,
  BsArrowDownUp,
  BsBeaker,
  BsCheckCircle,
  BsCloudArrowUp,
  BsCodeSlash,
  BsDownload,
  BsExclamationTriangle,
  BsEye,
  BsKey,
  BsShieldCheck,
} from "react-icons/bs";
import { apiFetch } from "../../lib/api";
import ModelNavigation from "../ModelNavigation";
import styles from "../../studio.module.css";
import {
  ADMIN_TOKEN_SESSION_KEY,
  LIFECYCLE,
  adminRequestOptions,
  canPromote,
  displayValue,
  experimentalInstallRequestOptions,
  lifecyclePosition,
  normalizeAnalysis,
  normalizeLabModels,
  normalizeModelId,
  normalizePackRegistry,
  packDifferences,
  packMutationKind,
  packMutationRequestOptions,
  shortRevision,
} from "./lab-state.mjs";

const STATUS_LABELS = {
  READY: "READY",
  EXPERIMENTAL: "EXPERIMENTAL",
  NEW: "NEW",
  UNSUPPORTED: "UNSUPPORTED",
};

const LIFECYCLE_LABELS = {
  DISCOVERED: "Découvert",
  ANALYZED: "Analysé",
  INSTALLED: "Installé",
  EXPERIMENTAL: "Expérimental",
  VALIDATED: "Validé",
  READY: "Prêt",
};

function statusClass(status) {
  return ({
    READY: styles.labStatusReady,
    EXPERIMENTAL: styles.labStatusExperimental,
    NEW: styles.labStatusNew,
    UNSUPPORTED: styles.labStatusUnsupported,
  })[status] || styles.labStatusUnknown;
}

function differenceClass(status) {
  return ({
    IDENTIQUE: styles.labDiffIdentical,
    "MODIFIÉ": styles.labDiffModified,
    "AJOUTÉ": styles.labDiffAdded,
    "SUPPRIMÉ": styles.labDiffRemoved,
  })[status] || styles.labDiffUnknown;
}

function DifferenceBadge({ status }) {
  return <span className={`${styles.labDifferenceBadge} ${differenceClass(status)}`}>{status || "INCONNU"}</span>;
}

function RegistryBadge({ status }) {
  return <span className={`${styles.labRegistryBadge} ${statusClass(status)}`}>{STATUS_LABELS[status] || status || "NEW"}</span>;
}

function ComparisonTable({ rows }) {
  return (
    <div className={styles.labTableScroll}>
      <table className={styles.labComparisonTable}>
        <thead><tr><th>Champ</th><th>Modèle analysé</th><th>Référence validée</th><th>Différence</th></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              <th scope="row">{row.label}</th>
              <td>{displayValue(row.candidate)}</td>
              <td>{displayValue(row.reference)}</td>
              <td><DifferenceBadge status={row.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Lifecycle({ status }) {
  const position = lifecyclePosition(status);
  return (
    <ol className={styles.labLifecycle} aria-label={`Cycle de validation : ${LIFECYCLE_LABELS[status] || status}`}>
      {LIFECYCLE.map((step, index) => (
        <li className={index < position ? styles.labLifecycleDone : index === position ? styles.labLifecycleCurrent : ""} key={step}>
          <span>{index < position ? "✓" : index + 1}</span><small>{LIFECYCLE_LABELS[step]}</small>
        </li>
      ))}
    </ol>
  );
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString("fr-FR");
}

function normalizedPackDiff(diff, index) {
  if (typeof diff === "string") return { key: `${index}-${diff}`, field: diff, current: "—", target: "—", status: "MODIFIÉ" };
  const entry = diff && typeof diff === "object" ? diff : {};
  return {
    key: `${index}-${entry.field || entry.path || entry.name || "diff"}`,
    field: entry.field || entry.path || entry.name || "Manifest",
    current: entry.current ?? entry.before ?? entry.reference,
    target: entry.target ?? entry.after ?? entry.candidate,
    status: entry.status || entry.kind || entry.change || "MODIFIÉ",
  };
}

export default function ModelLabPage() {
  const [modelInput, setModelInput] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [revision, setRevision] = useState("");
  const [models, setModels] = useState([]);
  const [packs, setPacks] = useState([]);
  const [packSelections, setPackSelections] = useState({});
  const [openPackDiffs, setOpenPackDiffs] = useState({});
  const [adminToken, setAdminToken] = useState(() => {
    if (typeof window === "undefined") return "";
    return window.sessionStorage.getItem(ADMIN_TOKEN_SESSION_KEY) || "";
  });
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadRegistry = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setLoading(true);
    const [modelsResult, packsResult] = await Promise.allSettled([
      apiFetch("/api/models/lab", { timeoutMs: 60_000, timeoutCode: "LAB_LIST_TIMEOUT" }),
      apiFetch("/api/model-packs/registry", { timeoutMs: 60_000, timeoutCode: "PACK_REGISTRY_TIMEOUT" }),
    ]);

    if (modelsResult.status === "fulfilled") setModels(normalizeLabModels(modelsResult.value));
    if (packsResult.status === "fulfilled") {
      const nextPacks = normalizePackRegistry(packsResult.value);
      setPacks(nextPacks);
      setPackSelections((current) => Object.fromEntries(nextPacks.map((pack) => [pack.id, current[pack.id] || pack.availableVersion || pack.currentVersion])));
    }

    const failures = [modelsResult, packsResult].filter((result) => result.status === "rejected");
    setError(failures.map((result) => result.reason?.message || "Registre indisponible").join(" "));
    if (!quiet) setLoading(false);
  }, []);

  useEffect(() => {
    const request = Promise.resolve().then(() => loadRegistry());
    return () => { void request; };
  }, [loadRegistry]);

  function updateAdminToken(value) {
    setAdminToken(value);
    if (value) window.sessionStorage.setItem(ADMIN_TOKEN_SESSION_KEY, value);
    else window.sessionStorage.removeItem(ADMIN_TOKEN_SESSION_KEY);
  }

  async function analyzeModel(event, explicitModelId) {
    event?.preventDefault();
    const modelId = normalizeModelId(explicitModelId || modelInput);
    if (!modelId) {
      setError("MODEL_ID_INVALID: utilisez le format owner/model ou une URL huggingface.co/owner/model.");
      return;
    }
    setAnalyzing(true);
    setError("");
    setNotice("");
    if (explicitModelId) setModelInput(modelId);
    try {
      const payload = await apiFetch("/api/models/lab/analyze", {
        method: "POST",
        timeoutMs: 120_000,
        timeoutCode: "LAB_ANALYSIS_TIMEOUT",
        body: JSON.stringify({ model_id: modelId }),
      });
      const nextAnalysis = normalizeAnalysis(payload, modelId);
      setAnalysis(nextAnalysis);
      setRevision(nextAnalysis.revision);
      setNotice("Analyse terminée. Le score est informatif : aucune promotion READY n'est automatique.");
      await loadRegistry({ quiet: true });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function installExperimental() {
    if (!analysis?.modelId || !revision.trim()) {
      setError("REVISION_REQUIRED: une révision ou un commit SHA doit être épinglé avant l'installation.");
      return;
    }
    const key = `install:${analysis.modelId}`;
    setBusy(key);
    setError("");
    setNotice("");
    try {
      await apiFetch("/api/models/lab/install", {
        timeoutMs: 120_000,
        timeoutCode: "LAB_INSTALL_TIMEOUT",
        ...experimentalInstallRequestOptions(analysis.modelId, revision.trim(), adminToken),
      });
      setNotice(`${analysis.modelId}@${shortRevision(revision)} est installé en EXPERIMENTAL. Aucun code distant n'est approuvé par cette action.`);
      await loadRegistry({ quiet: true });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  }

  async function promote(model) {
    const key = `promote:${model.labId}`;
    setBusy(key);
    setError("");
    setNotice("");
    try {
      await apiFetch(`/api/models/lab/${encodeURIComponent(model.labId)}/promote`, {
        method: "POST",
        timeoutMs: 60_000,
        timeoutCode: "LAB_PROMOTION_TIMEOUT",
        ...adminRequestOptions(adminToken),
      });
      setNotice(`${model.id}@${shortRevision(model.revision)} a été validé et promu explicitement.`);
      await loadRegistry({ quiet: true });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  }

  async function mutatePack(pack, action, version = "") {
    const key = `${action}:${pack.id}`;
    setBusy(key);
    setError("");
    setNotice("");
    try {
      await apiFetch(`/api/model-packs/${encodeURIComponent(pack.id)}/${action}`, {
        timeoutMs: 60_000,
        timeoutCode: `PACK_${action.toUpperCase()}_TIMEOUT`,
        ...packMutationRequestOptions(action, version, adminToken),
      });
      setNotice(action === "publish" ? `${pack.id} a été publié explicitement.` : `${pack.id} utilise maintenant la version ${version}.`);
      await loadRegistry({ quiet: true });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  }

  const modelCounts = useMemo(() => Object.fromEntries(["READY", "EXPERIMENTAL", "NEW", "UNSUPPORTED"].map((status) => [status, models.filter((model) => model.registryStatus === status).length])), [models]);

  return (
    <div className={styles.page}>
      <ModelNavigation active="lab" />
      <header className={styles.pageHeading}>
        <div><h1><BsBeaker /> VidioAI Lab</h1><p>Analysez un modèle Hugging Face, épinglez sa révision et validez-le sans promotion implicite.</p></div>
        <button className={styles.secondaryButton} disabled={loading} onClick={() => void loadRegistry()}><BsArrowClockwise /> Actualiser</button>
      </header>

      <section className={styles.labAdminBar} aria-label="Authentification administrateur">
        <div><BsKey /><span><strong>Actions administrateur</strong><small>Le jeton reste dans cette session du navigateur uniquement.</small></span></div>
        <label><span>Jeton admin</span><input autoComplete="off" type="password" value={adminToken} onChange={(event) => updateAdminToken(event.target.value)} placeholder="Bearer token" /></label>
      </section>

      {error && <div className={styles.errorBanner} role="alert"><BsExclamationTriangle /><span>{error}</span></div>}
      {notice && <div className={styles.successBanner} role="status"><BsCheckCircle /><span>{notice}</span></div>}

      <section className={styles.labAnalyzePanel}>
        <div className={styles.labSectionHeading}><div><span className={styles.eyebrow}>MODEL REGISTRY</span><h2>Analyser et comparer</h2><p>VidioAI lit les métadonnées publiques sans exécuter le code distant du repository.</p></div><BsCodeSlash /></div>
        <form className={styles.labAnalyzeForm} onSubmit={(event) => void analyzeModel(event)}>
          <label><span>Repository Hugging Face</span><input value={modelInput} onChange={(event) => setModelInput(event.target.value)} placeholder="owner/model" autoComplete="off" /></label>
          <button className={styles.primaryButton} disabled={analyzing} type="submit"><BsEye /> {analyzing ? "Analyse…" : "Analyser"}</button>
        </form>

        {analysis && <div className={styles.labAnalysisResult}>
          <div className={styles.labAnalysisSummary}>
            <div><span>Modèle</span><strong>{analysis.modelId}</strong><RegistryBadge status={analysis.registryStatus} /></div>
            <div><span>Famille / pack le plus proche</span><strong>{analysis.family}</strong><small>{analysis.closestPack} · v{analysis.closestPackVersion}</small></div>
            <div><span>Modèle validé de référence</span><strong>{analysis.closestModel}</strong><small>Comparaison structurelle, pas une validation.</small></div>
            <div><span>Similarité / risque</span><strong>{analysis.similarity === null ? "—" : `${analysis.similarity.toFixed(0)} %`} / {analysis.riskLabel || (analysis.risk === null ? "—" : `${analysis.risk.toFixed(0)} %`)}</strong><small>Informatif uniquement</small></div>
          </div>
          <div className={styles.warningBanner}><strong>Aucune promotion automatique.</strong> {"Un score élevé ne rend jamais ce modèle READY ; l'installation reste EXPERIMENTAL jusqu'à validation admin."}</div>
          <ComparisonTable rows={analysis.differences} />
          {analysis.candidatePack && <details className={styles.advancedPanel}><summary>ModelPack Candidate proposé</summary><pre className={styles.labManifest}>{JSON.stringify(analysis.candidatePack, null, 2)}</pre></details>}
          <div className={styles.labInstallBar}>
            <label><span>Révision / commit SHA épinglé</span><input value={revision} onChange={(event) => setRevision(event.target.value)} placeholder="Commit SHA obligatoire" /></label>
            <button className={styles.primaryButton} title={!adminToken ? "Jeton administrateur requis" : ""} disabled={!adminToken || !revision.trim() || busy === `install:${analysis.modelId}`} onClick={() => void installExperimental()}><BsDownload /> {busy === `install:${analysis.modelId}` ? "Installation…" : "Installer en EXPERIMENTAL"}</button>
          </div>
        </div>}
      </section>

      <section className={styles.labRegistrySection}>
        <div className={styles.labSectionHeading}><div><span className={styles.eyebrow}>VALIDATION</span><h2>Registre des modèles Lab</h2><p>Chaque révision reste épinglée ; une nouvelle révision doit être comparée explicitement.</p></div><div className={styles.labCounters}>{Object.entries(modelCounts).map(([status, count]) => <span key={status}>{status} <strong>{count}</strong></span>)}</div></div>
        {loading ? <div className={styles.stateCard}>Lecture du registre Lab…</div> : models.length ? <div className={styles.labModelGrid}>
          {models.map((model) => <article className={styles.labModelCard} key={`${model.id}@${model.revision}`}>
            <div className={styles.labCardHeading}><div><h3>{model.id}</h3><small>{model.family}</small></div><RegistryBadge status={model.registryStatus} /></div>
            <Lifecycle status={model.lifecycle} />
            <dl className={styles.compactDetails}>
              <div><dt>Révision installée</dt><dd title={model.revision}>{shortRevision(model.revision)}</dd></div>
              <div><dt>ModelPack</dt><dd>{model.packId} · v{model.packVersion}</dd></div>
              <div><dt>Workflow</dt><dd>v{model.workflowVersion}</dd></div>
              <div><dt>Validé le</dt><dd>{formatDate(model.validatedAt)}</dd></div>
            </dl>
            {model.hasNewRevision && <div className={styles.labRevisionNotice}><div><strong>Nouvelle révision disponible</strong><span title={model.availableRevision}>{shortRevision(model.availableRevision)}</span></div><button className={styles.secondaryButton} disabled={analyzing} onClick={() => void analyzeModel(null, model.id)}><ArrowDownUpIcon /> Comparer</button></div>}
            <button className={styles.primaryButton} title={!adminToken ? "Jeton administrateur requis" : !canPromote(model) ? "Le modèle doit être installé et expérimental" : ""} disabled={!adminToken || !canPromote(model) || busy === `promote:${model.labId}`} onClick={() => void promote(model)}><BsShieldCheck /> {busy === `promote:${model.labId}` ? "Promotion…" : "Valider / Promouvoir"}</button>
          </article>)}
        </div> : <div className={styles.stateCard}>Aucun modèle analysé dans le Lab.</div>}
      </section>

      <section className={styles.labRegistrySection}>
        <div className={styles.labSectionHeading}><div><span className={styles.eyebrow}>MODELPACK REGISTRY</span><h2>Versions et publication</h2><p>Mettez à jour un pack indépendamment de VidioAI ou revenez vers une version conservée.</p></div><BsCloudArrowUp /></div>
        {loading ? <div className={styles.stateCard}>Lecture des ModelPacks…</div> : packs.length ? <div className={styles.labPackGrid}>
          {packs.map((pack) => {
            const selectedVersion = packSelections[pack.id] || pack.availableVersion || pack.currentVersion;
            const differences = packDifferences(pack, selectedVersion).map(normalizedPackDiff);
            const isCurrent = selectedVersion === pack.currentVersion;
            const mutation = packMutationKind(pack, selectedVersion);
            return <article className={styles.labPackCard} key={pack.id}>
              <div className={styles.labCardHeading}><div><h3>{pack.id}</h3><small>{pack.family}</small></div><RegistryBadge status={pack.status} /></div>
              <dl className={styles.compactDetails}>
                <div><dt>ModelPack actuel</dt><dd>v{pack.currentVersion || "—"}</dd></div>
                <div><dt>Nouvelle version</dt><dd>{pack.availableVersion ? `v${pack.availableVersion}` : "Aucune"}</dd></div>
                <div><dt>SHA256</dt><dd title={pack.sha256}>{shortRevision(pack.sha256)}</dd></div>
                <div><dt>VidioAI minimum</dt><dd>{pack.minimumVidioaiVersion}</dd></div>
                <div><dt>Workflow</dt><dd>v{pack.workflowVersion}</dd></div>
                <div><dt>Versions conservées</dt><dd>{pack.versions.length}</dd></div>
              </dl>
              <label className={styles.labVersionSelect}><span>Version cible</span><select value={selectedVersion} onChange={(event) => setPackSelections((current) => ({ ...current, [pack.id]: event.target.value }))}>{pack.versions.map((version) => <option value={version.version} key={version.version}>{version.version}{version.version === pack.currentVersion ? " (actuelle)" : ""}</option>)}</select></label>
              <button className={styles.secondaryButton} onClick={() => setOpenPackDiffs((current) => ({ ...current, [pack.id]: !current[pack.id] }))}><BsArrowDownUp /> {openPackDiffs[pack.id] ? "Masquer les différences" : "Voir les différences"}</button>
              {openPackDiffs[pack.id] && <div className={styles.labPackDiffs}>{differences.length ? differences.map((diff) => <div key={diff.key}><span>{diff.field}</span><code>{displayValue(diff.current)} → {displayValue(diff.target)}</code><DifferenceBadge status={diff.status} /></div>) : <p>Aucune différence de manifeste pour cette version.</p>}</div>}
              <div className={styles.labPackActions}>
                <button className={styles.primaryButton} title={!adminToken ? "Jeton administrateur requis" : ""} disabled={!adminToken || isCurrent || mutation !== "update" || busy === `update:${pack.id}`} onClick={() => void mutatePack(pack, "update", selectedVersion)}>Mettre à jour</button>
                <button className={styles.secondaryButton} title={!adminToken ? "Jeton administrateur requis" : ""} disabled={!adminToken || isCurrent || mutation !== "rollback" || busy === `rollback:${pack.id}`} onClick={() => void mutatePack(pack, "rollback", selectedVersion)}>Rollback</button>
                <button className={styles.secondaryButton} title={!adminToken ? "Jeton administrateur requis" : ""} disabled={!adminToken || busy === `publish:${pack.id}`} onClick={() => void mutatePack(pack, "publish")}>Publier</button>
              </div>
            </article>;
          })}
        </div> : <div className={styles.stateCard}>Aucun ModelPack publié dans le registre.</div>}
      </section>
    </div>
  );
}

function ArrowDownUpIcon() {
  return <BsArrowDownUp aria-hidden="true" />;
}
