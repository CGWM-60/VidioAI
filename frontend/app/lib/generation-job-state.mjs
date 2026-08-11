export const TERMINAL_JOB_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function generationFromJob(currentGeneration, job) {
  if (job?.result?.generation) return job.result.generation;
  if (!currentGeneration || !job) return currentGeneration || null;
  return {
    ...currentGeneration,
    progress: Number.isFinite(job.progress) ? job.progress : currentGeneration.progress,
    job_status: job.status,
    job_stage: job.stage,
  };
}

export function reconcileGenerationAfterMissedEvent(currentGeneration, fetchedJob) {
  return generationFromJob(currentGeneration, fetchedJob);
}
