export const CLOUD_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function cloudRestorePayload(models) {
  return {
    models: models.map(({ repository, revision }) => ({ repository, revision })),
  };
}

export function cloudJobPresentation(job) {
  const status = String(job?.status || "queued").toLowerCase();
  return {
    status,
    restoring: ["queued", "dispatching", "running", "saving_output"].includes(status),
    completed: status === "completed",
    failed: status === "failed",
    progress: status === "failed" ? null : Number(job?.progress || 0),
    errorCode: job?.error?.code || "JOB_FAILED",
    errorMessage: job?.error?.message || job?.message || "La restauration a échoué.",
  };
}

export function restoredModelHref(job) {
  const view = cloudJobPresentation(job);
  const repository = job?.result?.repository || job?.model_id;
  return view.completed && repository
    ? `/models/installed?model=${encodeURIComponent(repository)}`
    : null;
}
