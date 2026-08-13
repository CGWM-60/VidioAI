export const GENERATION_PRESETS = ["FAST", "BALANCED", "QUALITY"];

export const VIDEO_PRESET_FALLBACKS = {
  FAST: "480p",
  BALANCED: "720p",
  QUALITY: "1080p",
};

export const IMAGE_PRESET_FALLBACKS = {
  FAST: "fast",
  BALANCED: "balanced",
  QUALITY: "quality",
};

const PARAMETER_LABELS = {
  seed: "Seed",
  steps: "Steps",
  cfg: "CFG",
  guidance_scale: "CFG",
  sampler: "Sampler",
  scheduler: "Scheduler",
  fps: "FPS",
  resolution: "Résolution",
  strength: "Strength",
  motion: "Motion",
};

function parameterSource(model) {
  const pack = model?.model_pack || model?.pack || {};
  return [
    model?.advanced_parameters,
    model?.supported_parameters,
    model?.parameter_support,
    model?.support?.advanced_parameters,
    model?.inputs?.advanced_parameters,
    pack.advanced_parameters,
    pack.supported_parameters,
    pack.parameter_support,
    pack.support?.advanced_parameters,
    pack.inputs?.advanced_parameters,
  ].find((candidate) => Array.isArray(candidate)
    ? candidate.length > 0
    : candidate && typeof candidate === "object"
      ? Object.keys(candidate).length > 0
      : Boolean(candidate)) || [];
}

export function advancedParameterDescriptors(model) {
  const source = parameterSource(model);
  const entries = Array.isArray(source)
    ? source.map((value) => [typeof value === "string" ? value : value?.key || value?.name || value?.id, value])
    : Object.entries(source || {});

  return entries.flatMap(([key, value]) => {
    if (!key || value === false || value?.supported === false) return [];
    const definition = typeof value === "object" && value !== null ? value : {};
    const options = definition.options || definition.values || definition.enum || null;
    return [{
      key,
      label: definition.label || PARAMETER_LABELS[key] || key.replaceAll("_", " "),
      type: definition.type || (options ? "select" : ["seed", "steps", "cfg", "guidance_scale", "fps", "strength", "motion"].includes(key) ? "number" : "text"),
      options: Array.isArray(options) ? options : null,
      min: definition.min,
      max: definition.max,
      step: definition.step,
      defaultValue: definition.default ?? definition.default_value ?? "",
    }];
  });
}

export function generationAdvancedPayload(descriptors, values) {
  return Object.fromEntries(descriptors.flatMap((descriptor) => {
    const raw = values[descriptor.key] ?? descriptor.defaultValue;
    if (raw === "" || raw === null || raw === undefined) return [];
    const parsed = descriptor.type === "number" ? Number(raw) : raw;
    return Number.isNaN(parsed) ? [] : [[descriptor.key, parsed]];
  }));
}
