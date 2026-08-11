import assert from "node:assert/strict";
import test from "node:test";

import { cloudJobPresentation, cloudRestorePayload, restoredModelHref } from "../app/models/cloud/cloud-state.mjs";
import { applyInstalledAction, installedModelAction } from "../app/models/installed/installed-state.mjs";

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

