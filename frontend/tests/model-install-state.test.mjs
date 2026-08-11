import assert from "node:assert/strict";
import test from "node:test";

import {
  dependencyView,
  INSTALL_STEPS,
  installationView,
  transferView,
} from "../app/models/install/state.mjs";

test("the installation timeline never claims CUDA load or inference", () => {
  const serialized = JSON.stringify(INSTALL_STEPS).toLowerCase();
  assert.equal(serialized.includes("cuda"), false);
  assert.equal(serialized.includes("inférence"), false);
  assert.deepEqual(INSTALL_STEPS.at(-1), [
    "installed",
    "Installé",
    "Le chargement runtime reste une action séparée",
  ]);
});

test("S3 transfer uses measured bytes and preserves current file", () => {
  const view = transferView({
    transfer: {
      direction: "upload",
      provider: "s3",
      bytes_total: 40 * 1073741824,
      bytes_transferred: 20 * 1073741824,
      files_total: 42,
      files_completed: 21,
      files_skipped: 3,
      current_file: "transformer/model-00002.safetensors",
      current_file_size: 10 * 1073741824,
      current_file_bytes: 4 * 1073741824,
      bytes_per_second: 178 * 1048576,
      eta_seconds: 138,
    },
  });
  assert.equal(view.percent, 50);
  assert.equal(view.current_file, "transformer/model-00002.safetensors");
  assert.equal(view.rateLabel, "178.0 Mo/s");
  assert.equal(view.etaLabel, "2 min 18 s");
});

test("a cache failure remains retryable after local installation completed", () => {
  const view = installationView({
    status: "completed",
    stage: "installed",
    progress: 100,
    cache_status: "CACHE_FAILED",
  });
  assert.equal(view.complete, true);
  assert.equal(view.canRetryCache, true);
  assert.equal(view.failed, false);
});

test("a failure at 20 percent preserves its code and exposes retry", () => {
  const view = installationView({
    status: "failed",
    stage: "failed",
    progress: 20,
    message: "Installation échouée à 20% · PIPELINE_CLASS_NOT_AVAILABLE",
  });
  assert.equal(view.terminal, true);
  assert.equal(view.failed, true);
  assert.equal(view.failureCode, "PIPELINE_CLASS_NOT_AVAILABLE");
  assert.equal(view.canRetry, true);
});

test("a completed installation is installed but does not imply READY", () => {
  const view = installationView({
    status: "completed",
    stage: "installed",
    progress: 100,
    message: "Modèle installé",
  });
  assert.equal(view.complete, true);
  assert.equal(view.currentIndex, INSTALL_STEPS.length - 1);
  assert.equal(INSTALL_STEPS[view.currentIndex][0], "installed");
});

test("dependency states remain explicit from download through failure", () => {
  for (const status of ["DOWNLOADING", "INSTALLING", "INSTALLED", "FAILED"]) {
    const view = dependencyView({
      dependency: {
        import_name: "bitsandbytes",
        package: "bitsandbytes",
        version: "0.49.2",
        status,
      },
    });
    assert.equal(view.status, status);
  }
  assert.equal(dependencyView({ dependency: { import_name: "bitsandbytes", status: "DOWNLOADING" } }).active, true);
  assert.equal(dependencyView({ dependency: { import_name: "bitsandbytes", status: "INSTALLING" } }).active, true);
  assert.equal(dependencyView({ dependency: { import_name: "bitsandbytes", status: "INSTALLED" } }).installed, true);
  assert.equal(dependencyView({ dependency: { import_name: "bitsandbytes", status: "FAILED" } }).failed, true);
});

test("successful auto-resolution stays visible and never invents MISSING_DEPENDENCY", () => {
  const job = {
    status: "completed",
    stage: "installed",
    progress: 100,
    message: "Modèle installé",
    dependency: {
      import_name: "bitsandbytes",
      package: "bitsandbytes",
      version: "0.49.2",
      status: "INSTALLED",
      automatic: true,
    },
  };
  assert.equal(installationView(job).failureCode, "");
  assert.equal(dependencyView(job).installed, true);
  assert.equal(dependencyView(job).automatic, true);
});
