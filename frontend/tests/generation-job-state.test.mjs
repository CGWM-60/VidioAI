import assert from "node:assert/strict";
import test from "node:test";

import {
  generationFromJob,
  reconcileGenerationAfterMissedEvent,
  TERMINAL_JOB_STATUSES,
} from "../app/lib/generation-job-state.mjs";

test("real job updates drive the UI and GET reconciliation recovers a missed completion", () => {
  let generation = { id: "generation-1", status: "queued", progress: 0 };
  generation = generationFromJob(generation, { status: "queued", stage: "queued", progress: 0 });
  generation = generationFromJob(generation, { status: "running", stage: "generating", progress: 47 });
  assert.equal(generation.progress, 47);
  assert.equal(generation.job_stage, "generating");

  const fetchedAfterLostWebSocket = {
    status: "completed",
    progress: 100,
    result: { generation: { id: "generation-1", status: "completed", progress: 100, output_asset_id: "asset-1" } },
  };
  generation = reconcileGenerationAfterMissedEvent(generation, fetchedAfterLostWebSocket);
  assert.equal(TERMINAL_JOB_STATUSES.has(fetchedAfterLostWebSocket.status), true);
  assert.equal(generation.status, "completed");
  assert.equal(generation.output_asset_id, "asset-1");
});
