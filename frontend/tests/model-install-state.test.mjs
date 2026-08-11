import assert from "node:assert/strict";
import test from "node:test";

import {
  INSTALL_STEPS,
  installationView,
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
