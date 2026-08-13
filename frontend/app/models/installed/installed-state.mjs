export function installedModelAction(model) {
  return model?.loaded || model?.state === "READY" ? "unload" : "load";
}

export function applyInstalledAction(model, action, succeeded) {
  if (!succeeded) return model;
  if (action === "load") return { ...model, state: "READY", loaded: true };
  if (action === "unload") return { ...model, state: "INSTALLED", loaded: false };
  return model;
}

export function runtimeMemoryValue(memory, ...keys) {
  for (const key of keys) {
    const value = memory?.[key];
    if (Number.isFinite(value)) return value;
  }
  return null;
}

export function runtimeUnloadPresentation(response) {
  if (!response) return null;
  const unloaded = Array.isArray(response.models_unloaded)
    ? response.models_unloaded.length
    : Number(response.models_unloaded || response.unloaded || 0);
  return {
    success: response.success !== false,
    unloaded: Number.isFinite(unloaded) ? unloaded : 0,
    before: response.before_memory || response.before || null,
    after: response.after_memory || response.after || null,
    message: response.message || "Les ressources runtime VidioAI ont été libérées.",
  };
}

export function installedModelMetadata(model) {
  const pack = model?.model_pack || model?.pack || null;
  return {
    modelPack: model?.model_pack_id || pack?.id || null,
    engine: model?.engine || pack?.engine || null,
    cloudBackup: model?.cloud_backup_status || model?.cache_status || null,
    residentBytes: runtimeMemoryValue(
      model?.telemetry || model?.memory || model,
      "model_resident_bytes",
      "vidioai_model_resident_bytes",
      "resident_bytes",
    ),
  };
}
