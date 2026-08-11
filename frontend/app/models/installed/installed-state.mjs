export function installedModelAction(model) {
  return model?.loaded || model?.state === "READY" ? "unload" : "load";
}

export function applyInstalledAction(model, action, succeeded) {
  if (!succeeded) return model;
  if (action === "load") return { ...model, state: "READY", loaded: true };
  if (action === "unload") return { ...model, state: "INSTALLED", loaded: false };
  return model;
}

