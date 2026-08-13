export const REGISTRY_STATUSES = new Set(["READY", "EXPERIMENTAL", "NEW", "UNSUPPORTED"]);
export const DIFFERENCE_STATUSES = new Set(["IDENTIQUE", "MODIFIÉ", "AJOUTÉ", "SUPPRIMÉ", "INCONNU"]);
export const LIFECYCLE = ["DISCOVERED", "ANALYZED", "INSTALLED", "EXPERIMENTAL", "VALIDATED", "READY"];
export const ADMIN_TOKEN_SESSION_KEY = "vidioai.lab.admin-token";

const COMPARISON_FIELDS = [
  ["architecture", "Architecture", ["architecture", "architectures"]],
  ["pipeline", "Pipeline", ["pipeline", "pipeline_class", "pipeline_tag"]],
  ["capabilities", "Capacités", ["capabilities", "tasks"]],
  ["configs", "Configurations", ["configs", "config", "configuration"]],
  ["vae", "VAE", ["vae", "vae_config"]],
  ["text_encoder", "Text encoder", ["text_encoder", "text_encoders"]],
  ["scheduler", "Scheduler", ["scheduler", "scheduler_config"]],
  ["files", "Fichiers", ["files", "siblings"]],
  ["revision", "Révision", ["revision", "commit_sha", "sha"]],
  ["model_size", "Taille du modèle", ["model_size", "size_bytes", "estimated_size_bytes"]],
];

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeToken(value) {
  return String(value || "")
    .trim()
    .toUpperCase()
    .replaceAll("É", "É")
    .replaceAll("-", "_")
    .replaceAll(" ", "_");
}

function normalizeDifferenceStatus(value) {
  const token = normalizeToken(value);
  const aliases = {
    SAME: "IDENTIQUE",
    IDENTICAL: "IDENTIQUE",
    MATCH: "IDENTIQUE",
    MATCHED: "IDENTIQUE",
    CHANGED: "MODIFIÉ",
    MODIFIED: "MODIFIÉ",
    DIFFERENT: "MODIFIÉ",
    ADDED: "AJOUTÉ",
    REMOVED: "SUPPRIMÉ",
    DELETED: "SUPPRIMÉ",
    UNKNOWN: "INCONNU",
  };
  const normalized = aliases[token] || token;
  return DIFFERENCE_STATUSES.has(normalized) ? normalized : "INCONNU";
}

function comparable(value) {
  if (value === undefined || value === null || value === "") return undefined;
  if (Array.isArray(value)) return [...value].map(comparable).sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
  if (typeof value === "object") {
    return Object.fromEntries(Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, comparable(item)]));
  }
  return value;
}

function inferredDifference(candidate, reference) {
  const left = comparable(candidate);
  const right = comparable(reference);
  if (left === undefined && right === undefined) return "INCONNU";
  if (left !== undefined && right === undefined) return "AJOUTÉ";
  if (left === undefined && right !== undefined) return "SUPPRIMÉ";
  return JSON.stringify(left) === JSON.stringify(right) ? "IDENTIQUE" : "MODIFIÉ";
}

function valueAt(source, aliases) {
  for (const alias of aliases) {
    if (source[alias] !== undefined && source[alias] !== null) return source[alias];
  }
  return undefined;
}

function differenceEntries(payload) {
  const root = asObject(payload);
  const comparison = root.comparison;
  const raw = Array.isArray(comparison)
    ? comparison
    : firstDefined(asObject(comparison).differences, root.differences, root.diff, []);

  if (Array.isArray(raw)) return raw.map((entry) => asObject(entry));
  return Object.entries(asObject(raw)).map(([field, entry]) => {
    if (entry && typeof entry === "object" && !Array.isArray(entry)) return { field, ...entry };
    return { field, status: entry };
  });
}

function canonicalField(value) {
  const token = String(value || "").trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  return COMPARISON_FIELDS.find(([, , aliases]) => aliases.includes(token))?.[0] || token;
}

export function normalizeComparison(payload) {
  const root = asObject(payload);
  const comparison = asObject(root.comparison);
  const candidate = asObject(firstDefined(root.model, root.metadata, root.candidate, root.analyzed_model, comparison.candidate));
  const closest = asObject(firstDefined(root.closest_model, root.closest, root.reference_model, comparison.reference));
  const reference = asObject(firstDefined(closest.metadata, closest.model, closest.reference, closest));
  const explicit = new Map(differenceEntries(root).map((entry) => [canonicalField(firstDefined(entry.field, entry.name, entry.key)), entry]));
  const standardKeys = new Set(COMPARISON_FIELDS.map(([key]) => key));

  const rows = COMPARISON_FIELDS.map(([key, label, aliases]) => {
    const entry = explicit.get(key) || {};
    const candidateValue = firstDefined(entry.candidate, entry.actual, entry.value, entry.left, valueAt(candidate, aliases));
    const referenceValue = firstDefined(entry.reference, entry.expected, entry.baseline, entry.right, valueAt(reference, aliases));
    const hasExplicitStatus = firstDefined(entry.status, entry.difference, entry.kind, entry.change) !== undefined;
    return {
      key,
      label,
      candidate: candidateValue,
      reference: referenceValue,
      status: hasExplicitStatus
        ? normalizeDifferenceStatus(firstDefined(entry.status, entry.difference, entry.kind, entry.change))
        : inferredDifference(candidateValue, referenceValue),
    };
  });

  for (const entry of explicit.values()) {
    const key = canonicalField(firstDefined(entry.field, entry.name, entry.key));
    if (!key || standardKeys.has(key)) continue;
    const candidateValue = firstDefined(entry.candidate, entry.actual, entry.value, entry.left);
    const referenceValue = firstDefined(entry.reference, entry.expected, entry.baseline, entry.right);
    rows.push({
      key,
      label: firstDefined(entry.label, entry.field, entry.name, key),
      candidate: candidateValue,
      reference: referenceValue,
      status: normalizeDifferenceStatus(firstDefined(entry.status, entry.difference, entry.kind, entry.change, inferredDifference(candidateValue, referenceValue))),
    });
  }

  return rows;
}

export function normalizeModelId(value) {
  let candidate = String(value || "").trim();
  if (!candidate) return "";

  try {
    const parsed = new URL(candidate);
    if (parsed.protocol === "hf:") {
      candidate = [parsed.hostname, ...parsed.pathname.split("/").filter(Boolean)].slice(0, 2).join("/");
    } else {
      if (!["huggingface.co", "www.huggingface.co"].includes(parsed.hostname.toLowerCase())) return "";
      const segments = parsed.pathname.split("/").filter(Boolean);
      candidate = segments.slice(0, 2).join("/");
    }
  } catch {
    candidate = candidate.replace(/^hf:\/\//i, "").split(/[?#]/, 1)[0].replace(/^\/+|\/+$/g, "");
  }

  const parts = candidate.split("/");
  if (parts.length !== 2 || parts.some((part) => !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(part) || part === "." || part === "..")) return "";
  return parts.join("/");
}

function normalizedRegistryStatus(root) {
  const explicit = normalizeToken(firstDefined(root.registry_status, root.model_status, root.registry?.status));
  if (REGISTRY_STATUSES.has(explicit)) return explicit;
  const lifecycle = normalizeToken(firstDefined(root.lifecycle, root.lifecycle_status, root.stage));
  if (lifecycle === "READY") return "READY";
  if (["INSTALLED", "EXPERIMENTAL", "VALIDATED"].includes(lifecycle)) return "EXPERIMENTAL";
  return root.supported === false || root.compatible === false ? "UNSUPPORTED" : "NEW";
}

function score(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, numeric <= 1 ? numeric * 100 : numeric));
}

export function normalizeAnalysis(payload, fallbackModelId = "") {
  const envelope = asObject(payload);
  const root = asObject(firstDefined(envelope.analysis, envelope.result, envelope));
  const model = asObject(firstDefined(root.model, root.metadata, root.candidate, root.analyzed_model));
  const closest = asObject(firstDefined(root.closest_model, root.closest, root.reference_model));
  const candidatePack = asObject(firstDefined(root.model_pack_candidate, root.candidate_pack, root.pack_candidate));
  const similarity = score(firstDefined(root.similarity_score, root.similarity, closest.similarity_score, closest.similarity));
  const rawRisk = firstDefined(root.risk_score, root.risk, closest.risk_score, closest.risk);
  const explicitRisk = score(rawRisk);

  return {
    raw: root,
    modelId: firstDefined(root.model_id, root.repository, model.model_id, model.repository, model.id, fallbackModelId) || "",
    revision: firstDefined(root.revision, root.commit_sha, root.sha, model.revision, model.commit_sha, model.sha) || "",
    registryStatus: normalizedRegistryStatus({ ...root, lifecycle: firstDefined(root.lifecycle, model.lifecycle) }),
    lifecycle: normalizeToken(firstDefined(root.lifecycle, root.lifecycle_status, root.stage, "ANALYZED")),
    family: firstDefined(root.family, closest.family, candidatePack.family) || "Famille inconnue",
    closestModel: firstDefined(closest.model_id, closest.repository, closest.id, root.closest_model, root.reference_model_id) || "Aucun modèle validé proche",
    closestPack: firstDefined(root.closest_pack_id, root.closest_pack, closest.model_pack_id, closest.pack_id, candidatePack.based_on_pack, candidatePack.base_pack_id, candidatePack.id) || "Aucun ModelPack connu",
    closestPackVersion: firstDefined(closest.model_pack_version, closest.pack_version, candidatePack.base_version, candidatePack.version) || "—",
    candidatePack: Object.keys(candidatePack).length ? candidatePack : null,
    similarity,
    risk: explicitRisk ?? (similarity === null ? null : 100 - similarity),
    riskLabel: typeof rawRisk === "string" && !Number.isFinite(Number(rawRisk)) ? normalizeToken(rawRisk) : "",
    differences: normalizeComparison(root),
  };
}

function normalizeLifecycle(value, registryStatus) {
  const token = normalizeToken(value);
  if (LIFECYCLE.includes(token)) return token;
  if (registryStatus === "READY") return "READY";
  if (registryStatus === "EXPERIMENTAL") return "EXPERIMENTAL";
  return "DISCOVERED";
}

export function normalizeLabModels(payload) {
  const root = asObject(payload);
  const items = Array.isArray(payload) ? payload : firstDefined(root.items, root.models, root.entries, root.results, []);
  const models = asArray(items).map((item) => {
    const model = asObject(item);
    const registryStatus = normalizedRegistryStatus(model);
    const revision = firstDefined(model.revision, model.commit_sha, model.sha, "") || "";
    const availableRevision = firstDefined(model.available_revision, model.latest_revision, model.new_revision, "") || "";
    const candidate = asObject(model.model_pack_candidate);
    return {
      ...model,
      labId: firstDefined(model.entry_id, model.id, "") || "",
      id: firstDefined(model.model_id, model.repository, model.repo_id, model.id, "") || "",
      revision,
      availableRevision,
      hasNewRevision: Boolean(model.new_revision_available || (availableRevision && revision && availableRevision !== revision)),
      registryStatus,
      lifecycle: normalizeLifecycle(firstDefined(model.lifecycle, model.lifecycle_status, model.stage, model.status), registryStatus),
      family: firstDefined(model.family, model.model_pack?.family, candidate.family, "Famille inconnue"),
      packId: firstDefined(model.model_pack_id, model.pack_id, model.closest_pack, model.model_pack?.id, candidate.id, "—"),
      packVersion: firstDefined(model.model_pack_version, model.pack_version, model.model_pack?.version, candidate.version, "—"),
      workflowVersion: firstDefined(model.workflow_version, model.workflow?.version, candidate.workflow_version, "—"),
      validatedAt: firstDefined(model.validated_at, model.validation?.validated_at, model.promotion?.validated_at, null),
      updatedAt: Number(firstDefined(model.updated_at, model.created_at, 0)) || 0,
    };
  });

  return models.map((model) => {
    if (model.availableRevision || !model.update_available) return model;
    const newer = models
      .filter((candidate) => candidate.id === model.id && candidate.revision !== model.revision && candidate.updatedAt > model.updatedAt)
      .sort((left, right) => right.updatedAt - left.updatedAt)[0];
    return newer ? { ...model, availableRevision: newer.revision, hasNewRevision: true } : model;
  });
}

export function lifecyclePosition(status) {
  const index = LIFECYCLE.indexOf(normalizeToken(status));
  return index < 0 ? 0 : index;
}

export function canPromote(model) {
  if (!model?.labId || !model.revision || model.registryStatus === "READY" || model.registryStatus === "UNSUPPORTED") return false;
  return lifecyclePosition(model.lifecycle) >= LIFECYCLE.indexOf("INSTALLED");
}

function normalizeVersion(value, fallbackManifest = null) {
  if (typeof value === "string") return { version: value, manifest: null, differences: [] };
  const item = asObject(value);
  return {
    ...item,
    version: firstDefined(item.version, item.id, "") || "",
    manifest: firstDefined(item.manifest, item.payload, fallbackManifest),
    differences: asArray(firstDefined(item.differences, item.diff, [])),
  };
}

function flatten(value, prefix = "", output = new Map()) {
  if (Array.isArray(value)) {
    output.set(prefix, value);
    return output;
  }
  if (!value || typeof value !== "object") {
    output.set(prefix, value);
    return output;
  }
  for (const [key, child] of Object.entries(value)) flatten(child, prefix ? `${prefix}.${key}` : key, output);
  return output;
}

export function compareManifests(current, target) {
  const before = flatten(asObject(current));
  const after = flatten(asObject(target));
  const keys = [...new Set([...before.keys(), ...after.keys()])].filter(Boolean).sort();
  return keys
    .map((key) => ({ field: key, current: before.get(key), target: after.get(key), status: inferredDifference(after.get(key), before.get(key)) }))
    .filter((entry) => entry.status !== "IDENTIQUE");
}

function compareVersion(left, right) {
  const parts = (value) => String(value || "").split(/[.+-]/).map((part) => /^\d+$/.test(part) ? Number(part) : part);
  const leftParts = parts(left);
  const rightParts = parts(right);
  for (let index = 0; index < Math.max(leftParts.length, rightParts.length); index += 1) {
    const leftPart = leftParts[index] ?? 0;
    const rightPart = rightParts[index] ?? 0;
    if (leftPart === rightPart) continue;
    if (typeof leftPart === "number" && typeof rightPart === "number") return leftPart - rightPart;
    return String(leftPart).localeCompare(String(rightPart), undefined, { numeric: true });
  }
  return 0;
}

function normalizePack(pack) {
    const currentObject = asObject(pack.current);
    const currentManifest = firstDefined(pack.manifest, currentObject.manifest, currentObject.payload, Object.keys(currentObject).length ? currentObject : pack);
    const currentVersion = String(firstDefined(pack.current_version, currentObject.version, pack.version, "") || "");
    const rawVersions = firstDefined(pack.versions, pack.available_versions, []);
    const versions = asArray(rawVersions).map((version) => normalizeVersion(version));
    if (currentVersion && !versions.some((version) => version.version === currentVersion)) {
      versions.unshift(normalizeVersion({ version: currentVersion, manifest: currentManifest }));
    }
    const explicitAvailable = normalizeVersion(firstDefined(pack.available, pack.latest, pack.available_version, {}));
    const available = explicitAvailable.version
      ? explicitAvailable
      : [...versions].reverse().find((version) => version.version && version.version !== currentVersion) || null;
    const selectedManifest = available?.manifest;
    const differences = asArray(firstDefined(pack.differences, pack.diff, available?.differences));

    return {
      ...pack,
      id: firstDefined(pack.pack_id, pack.id, pack.manifest?.id, "") || "",
      family: firstDefined(pack.family, pack.manifest?.family, "Famille inconnue"),
      currentVersion,
      currentManifest,
      availableVersion: available?.version || "",
      availableManifest: selectedManifest,
      versions,
      differences: differences.length ? differences : compareManifests(currentManifest, selectedManifest),
      sha256: firstDefined(pack.sha256, currentObject.sha256, pack.manifest_sha256, "—"),
      minimumVidioaiVersion: firstDefined(pack.minimum_vidioai_version, pack.min_vidioai_version, pack.manifest?.minimum_vidioai_version, "—"),
      workflowVersion: firstDefined(pack.workflow_version, pack.workflow?.version, pack.manifest?.workflow?.version, "—"),
      status: normalizeToken(firstDefined(pack.status, currentObject.status, pack.manifest?.status, "READY")),
    };
}

export function normalizePackRegistry(payload) {
  const root = asObject(payload);
  const items = asArray(Array.isArray(payload) ? payload : firstDefined(root.items, root.packs, root.entries, root.results, []));
  const manifests = asArray(root.manifests);
  const manifestsByVersion = new Map(manifests.map((manifest) => {
    const item = asObject(manifest);
    const id = firstDefined(item.pack_id, item.id);
    return [`${id || ""}\u0000${item.version || ""}`, item];
  }));
  const joinedItems = items.map((record) => {
    const item = asObject(record);
    const id = firstDefined(item.pack_id, item.id);
    const versionedManifest = manifestsByVersion.get(`${id || ""}\u0000${item.version || ""}`);
    if (!versionedManifest) return item;
    return {
      ...item,
      manifest: firstDefined(item.manifest, item.pack, versionedManifest.pack),
    };
  });
  const flatVersionRecords = joinedItems.length > 0 && joinedItems.every((item) => item && typeof item === "object" && "version" in item && !("versions" in item) && !("current_version" in item));
  if (!flatVersionRecords) return joinedItems.map((item) => normalizePack(asObject(item)));

  const grouped = new Map();
  for (const record of joinedItems) {
    const id = firstDefined(record.pack_id, record.id);
    if (!id) continue;
    grouped.set(id, [...(grouped.get(id) || []), record]);
  }

  return [...grouped.entries()].map(([id, records]) => {
    const ordered = [...records].sort((left, right) => compareVersion(left.version, right.version));
    const current = ordered.find((record) => record.active) || ordered.at(-1);
    const available = [...ordered].reverse().find((record) => compareVersion(record.version, current.version) > 0) || null;
    return normalizePack({
      id,
      family: current.family,
      current_version: current.version,
      current: { ...current, manifest: firstDefined(current.manifest, current.pack) },
      available: available ? { ...available, manifest: firstDefined(available.manifest, available.pack) } : null,
      versions: ordered.map((record) => ({ ...record, manifest: firstDefined(record.manifest, record.pack) })),
      sha256: current.sha256,
      min_vidioai_version: current.min_vidioai_version,
      workflow_version: current.workflow_version,
      status: current.status,
    });
  });
}

export function packVersion(pack, version) {
  return pack?.versions?.find((item) => item.version === version) || null;
}

export function packDifferences(pack, version) {
  const target = packVersion(pack, version);
  if (!target) return pack?.differences || [];
  if (target.differences?.length) return target.differences;
  return compareManifests(pack.currentManifest, target.manifest);
}

export function packMutationKind(pack, version) {
  if (!pack || !version || version === pack.currentVersion) return null;
  return compareVersion(version, pack.currentVersion) < 0 ? "rollback" : "update";
}

export function displayValue(value) {
  if (value === undefined || value === null || value === "") return "Inconnu";
  if (typeof value === "number" && Number.isFinite(value) && value >= 1073741824) return `${(value / 1073741824).toFixed(1)} Go`;
  if (Array.isArray(value)) return value.length ? value.map(displayValue).join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function shortRevision(value) {
  const revision = String(value || "");
  return revision.length > 14 ? `${revision.slice(0, 12)}…` : revision || "—";
}

export function adminRequestOptions(token) {
  const value = String(token || "").trim();
  return value ? { headers: { Authorization: `Bearer ${value}` } } : { headers: {} };
}

export function experimentalInstallRequestOptions(modelId, revision, token) {
  return {
    method: "POST",
    ...adminRequestOptions(token),
    body: JSON.stringify({ model_id: modelId, revision }),
  };
}

export function packMutationRequestOptions(action, version, token) {
  return {
    method: "POST",
    ...adminRequestOptions(token),
    body: JSON.stringify(action === "publish" ? {} : { version }),
  };
}
