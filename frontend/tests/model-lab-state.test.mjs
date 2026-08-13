import assert from "node:assert/strict";
import test from "node:test";

import {
  ADMIN_TOKEN_SESSION_KEY,
  adminRequestOptions,
  canPromote,
  compareManifests,
  experimentalInstallRequestOptions,
  lifecyclePosition,
  normalizeAnalysis,
  normalizeLabModels,
  normalizeModelId,
  normalizePackRegistry,
  packDifferences,
  packMutationKind,
  packMutationRequestOptions,
} from "../app/models/lab/lab-state.mjs";

test("normalizes only a strict Hugging Face owner/model identity", () => {
  assert.equal(normalizeModelId(" owner/model "), "owner/model");
  assert.equal(normalizeModelId("https://huggingface.co/owner/model/tree/main?x=1"), "owner/model");
  assert.equal(normalizeModelId("hf://org-a/model_2"), "org-a/model_2");
  assert.equal(normalizeModelId("https://example.com/owner/model"), "");
  assert.equal(normalizeModelId("owner/model/extra"), "");
  assert.equal(normalizeModelId("../model"), "");
});

test("a high similarity analysis stays NEW and never becomes READY", () => {
  const analysis = normalizeAnalysis({
    model: {
      repository: "owner/new-model",
      revision: "abcdef0123456789",
      architecture: "WanPipeline",
      capabilities: ["TEXT_TO_VIDEO"],
      size_bytes: 64 * 1024 ** 3,
    },
    closest_model: {
      repository: "known/validated-model",
      architecture: "WanPipeline",
      capabilities: ["TEXT_TO_VIDEO"],
      size_bytes: 60 * 1024 ** 3,
      model_pack_id: "wan22-t2v-v1",
      model_pack_version: "1.0.0",
      family: "wan-2.2",
    },
    similarity_score: 0.99,
  });

  assert.equal(analysis.registryStatus, "NEW");
  assert.equal(analysis.similarity, 99);
  assert.equal(analysis.risk, 1);
  assert.equal(analysis.closestPack, "wan22-t2v-v1");
  assert.equal(analysis.differences.find((row) => row.key === "architecture").status, "IDENTIQUE");
  assert.equal(analysis.differences.find((row) => row.key === "model_size").status, "MODIFIÉ");
});

test("comparison accepts object maps and translates backend aliases", () => {
  const analysis = normalizeAnalysis({
    model_id: "owner/model",
    registry_status: "EXPERIMENTAL",
    comparison: {
      differences: {
        pipeline: { status: "same", actual: "PipelineA", expected: "PipelineA" },
        vae: { status: "changed", actual: "vae-2", expected: "vae-1" },
        scheduler: { status: "added", actual: "Euler" },
        files: { status: "removed", expected: ["old.bin"] },
      },
    },
  });

  assert.equal(analysis.registryStatus, "EXPERIMENTAL");
  assert.equal(analysis.differences.find((row) => row.key === "pipeline").status, "IDENTIQUE");
  assert.equal(analysis.differences.find((row) => row.key === "vae").status, "MODIFIÉ");
  assert.equal(analysis.differences.find((row) => row.key === "scheduler").status, "AJOUTÉ");
  assert.equal(analysis.differences.find((row) => row.key === "files").status, "SUPPRIMÉ");
});

test("normalizes Lab lifecycle, pinned revisions, and promotion eligibility", () => {
  const [experimental, ready, unsupported] = normalizeLabModels({ items: [
    { id: "d11904e5-914b-4d90-a43c-126d8cb11445", repository: "owner/experimental", revision: "sha-1", lifecycle: "EXPERIMENTAL", latest_revision: "sha-2" },
    { id: "dd466791-c1f0-48c1-833f-437fe70c4d46", model_id: "owner/ready", commit_sha: "sha-ready", lifecycle: "READY" },
    { id: "f6c709e7-133a-49cd-9020-cf66da2f177e", model_id: "owner/nope", revision: "sha-nope", registry_status: "UNSUPPORTED", lifecycle: "INSTALLED" },
  ] });

  assert.equal(experimental.labId, "d11904e5-914b-4d90-a43c-126d8cb11445");
  assert.equal(experimental.id, "owner/experimental");
  assert.equal(experimental.lifecycle, "EXPERIMENTAL");
  assert.equal(experimental.registryStatus, "EXPERIMENTAL");
  assert.equal(experimental.hasNewRevision, true);
  assert.equal(experimental.availableRevision, "sha-2");
  assert.equal(canPromote(experimental), true);
  assert.equal(ready.lifecycle, "READY");
  assert.equal(ready.registryStatus, "READY");
  assert.equal(canPromote(ready), false);
  assert.equal(canPromote(unsupported), false);
  assert.equal(lifecyclePosition("validated"), 4);
});

test("normalizes versioned ModelPacks and computes manifest differences", () => {
  const [pack] = normalizePackRegistry({ packs: [{
    pack_id: "wan22-t2v-v1",
    family: "wan-2.2",
    current_version: "1.0.0",
    manifest: { workflow: { version: "1" }, steps: 20 },
    available: { version: "1.1.0", manifest: { workflow: { version: "2" }, steps: 24 } },
    versions: [
      { version: "0.9.0", manifest: { workflow: { version: "1" }, steps: 18 } },
      { version: "1.0.0", manifest: { workflow: { version: "1" }, steps: 20 } },
      { version: "1.1.0", manifest: { workflow: { version: "2" }, steps: 24 } },
    ],
  }] });

  assert.equal(pack.id, "wan22-t2v-v1");
  assert.equal(pack.currentVersion, "1.0.0");
  assert.equal(pack.availableVersion, "1.1.0");
  const differences = packDifferences(pack, "1.1.0");
  assert.deepEqual(differences.map((entry) => entry.field), ["steps", "workflow.version"]);
  assert.ok(differences.every((entry) => entry.status === "MODIFIÉ"));
  assert.equal(compareManifests({ engine: "diffusers" }, { engine: "comfyui" })[0].status, "MODIFIÉ");
  assert.equal(packMutationKind(pack, "0.9.0"), "rollback");
  assert.equal(packMutationKind(pack, "1.1.0"), "update");
  assert.equal(packMutationKind(pack, "1.0.0"), null);
});

test("groups the backend flat version registry into one selectable pack", () => {
  const [pack] = normalizePackRegistry({ packs: [
    { id: "flux-t2i-v1", family: "flux", version: "1.0.0", active: true, sha256: "a" },
    { id: "flux-t2i-v1", family: "flux", version: "1.1.0", active: false, sha256: "b" },
    { id: "flux-t2i-v1", family: "flux", version: "0.9.0", active: false, sha256: "c" },
  ] });

  assert.equal(pack.id, "flux-t2i-v1");
  assert.equal(pack.currentVersion, "1.0.0");
  assert.equal(pack.availableVersion, "1.1.0");
  assert.deepEqual(pack.versions.map((version) => version.version), ["0.9.0", "1.0.0", "1.1.0"]);
});

test("joins backend PackVersionRecords with VersionedPackManifests before comparing packs", () => {
  const [pack] = normalizePackRegistry({
    packs: [
      {
        id: "wan22-t2v-v1",
        family: "wan-2.2",
        version: "1.0.0",
        sha256: "old-sha",
        min_vidioai_version: "0.1.0",
        workflow_version: "workflow-v1",
        workflows: [],
        active: true,
        source: "bundled",
        created_at: 1,
        published_at: null,
      },
      {
        id: "wan22-t2v-v1",
        family: "wan-2.2",
        version: "1.1.0",
        sha256: "new-sha",
        min_vidioai_version: "0.1.0",
        workflow_version: "workflow-v2",
        workflows: [],
        active: false,
        source: "s3",
        created_at: 2,
        published_at: 3,
      },
    ],
    manifests: [
      {
        schema_version: 1,
        id: "wan22-t2v-v1",
        version: "1.0.0",
        sha256: "old-sha",
        min_vidioai_version: "0.1.0",
        workflow_version: "workflow-v1",
        workflows: [],
        created_at: 1,
        pack: { id: "wan22-t2v-v1", engine: "diffusers", defaults: { steps: 20 } },
      },
      {
        schema_version: 1,
        id: "wan22-t2v-v1",
        version: "1.1.0",
        sha256: "new-sha",
        min_vidioai_version: "0.1.0",
        workflow_version: "workflow-v2",
        workflows: [],
        created_at: 2,
        pack: { id: "wan22-t2v-v1", engine: "comfyui", defaults: { steps: 24 } },
      },
    ],
  });

  assert.deepEqual(pack.currentManifest, { id: "wan22-t2v-v1", engine: "diffusers", defaults: { steps: 20 } });
  assert.deepEqual(pack.availableManifest, { id: "wan22-t2v-v1", engine: "comfyui", defaults: { steps: 24 } });
  assert.deepEqual(
    packDifferences(pack, "1.1.0").map(({ field, status }) => ({ field, status })),
    [
      { field: "defaults.steps", status: "MODIFIÉ" },
      { field: "engine", status: "MODIFIÉ" },
    ],
  );
});

test("admin authentication is session-scoped and isolated from request bodies", () => {
  const secret = "top-secret-token";
  const options = adminRequestOptions(secret);

  assert.equal(ADMIN_TOKEN_SESSION_KEY, "vidioai.lab.admin-token");
  assert.deepEqual(options, { headers: { Authorization: `Bearer ${secret}` } });
  assert.equal(options.body, undefined);
  assert.equal(options.url, undefined);
  assert.equal(JSON.stringify({ model_id: "owner/model", revision: "sha" }).includes(secret), false);
  assert.deepEqual(adminRequestOptions(""), { headers: {} });
});

test("experimental install authenticates without leaking the admin token", () => {
  const secret = "lab-admin-secret";
  const options = experimentalInstallRequestOptions("owner/model", "commit-sha", secret);

  assert.equal(options.headers.Authorization, `Bearer ${secret}`);
  assert.deepEqual(JSON.parse(options.body), { model_id: "owner/model", revision: "commit-sha" });
  assert.equal(options.body.includes(secret), false);
  assert.equal(options.url, undefined);
});

test("ModelPack publication sends the empty JSON object required by Axum", () => {
  const secret = "pack-admin-secret";
  const publish = packMutationRequestOptions("publish", "", secret);
  const update = packMutationRequestOptions("update", "1.2.0", secret);

  assert.deepEqual(JSON.parse(publish.body), {});
  assert.deepEqual(JSON.parse(update.body), { version: "1.2.0" });
  assert.equal(publish.headers.Authorization, `Bearer ${secret}`);
  assert.equal(publish.body.includes(secret), false);
  assert.equal(publish.url, undefined);
});
