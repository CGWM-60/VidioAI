import assert from "node:assert/strict";
import test from "node:test";

import { cloudJobPresentation, cloudRestorePayload, restoredModelHref } from "../app/models/cloud/cloud-state.mjs";
import {
  applyInstalledAction,
  installedModelAction,
  installedModelMetadata,
  runtimeUnloadPresentation,
} from "../app/models/installed/installed-state.mjs";

test("cloud selection posts repository and revision only", () => {
  assert.deepEqual(cloudRestorePayload([{ repository: "owner/model", revision: "abc", manifest_uri: "s3://secret" }]), {
    models: [{ repository: "owner/model", revision: "abc" }],
  });
});

test("cloud jobs render running, completed and failed without failed 100 percent", () => {
  assert.equal(cloudJobPresentation({ status: "running", progress: 66 }).restoring, true);
  assert.equal(cloudJobPresentation({ status: "completed", progress: 100 }).completed, true);
  const failed = cloudJobPresentation({ status: "failed", progress: 100, error: { code: "S3_ERROR", message: "boom" } });
  assert.equal(failed.failed, true);
  assert.equal(failed.progress, null);
  assert.equal(failed.errorCode, "S3_ERROR");
});

test("an INSTALLED model exposes the load action", () => {
  assert.equal(installedModelAction({ state: "INSTALLED", loaded: false }), "load");
});

test("a READY model exposes the unload action", () => {
  assert.equal(installedModelAction({ state: "READY", loaded: true }), "unload");
});

test("successful load becomes READY in the UI contract", () => {
  assert.deepEqual(applyInstalledAction({ state: "INSTALLED", loaded: false }, "load", true), { state: "READY", loaded: true });
});

test("successful unload becomes INSTALLED in the UI contract", () => {
  assert.deepEqual(applyInstalledAction({ state: "READY", loaded: true }, "unload", true), { state: "INSTALLED", loaded: false });
});

test("completed cloud restoration links to the installed inventory", () => {
  assert.equal(restoredModelHref({ status: "completed", model_id: "owner/model" }), "/models/installed?model=owner%2Fmodel");
});

test("global runtime unload accepts zero loaded models and memory diagnostics", () => {
  assert.deepEqual(runtimeUnloadPresentation({
    success: true,
    models_unloaded: 0,
    before_memory: { nvml_gpu_used_bytes: 12 },
    after_memory: { nvml_gpu_used_bytes: 0 },
    message: "Aucun modèle déclaré; caches nettoyés.",
  }), {
    success: true,
    unloaded: 0,
    before: { nvml_gpu_used_bytes: 12 },
    after: { nvml_gpu_used_bytes: 0 },
    message: "Aucun modèle déclaré; caches nettoyés.",
  });
});

test("installed inventory renders optional ModelPack, engine and cloud state", () => {
  assert.deepEqual(installedModelMetadata({
    model_pack: { id: "wan22-i2v", engine: "COMFYUI" },
    cloud_backup_status: "COMPLETED",
    telemetry: { model_resident_bytes: 64 },
  }), {
    modelPack: "wan22-i2v",
    engine: "COMFYUI",
    cloudBackup: "COMPLETED",
    residentBytes: 64,
  });
});
