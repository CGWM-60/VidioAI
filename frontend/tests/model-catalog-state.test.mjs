import assert from "node:assert/strict";
import test from "node:test";

import {
  accessStatus,
  CATALOG_TIMEOUT_MS,
  MODEL_PREFLIGHT_TIMEOUT_MS,
} from "../app/models/catalog-state.mjs";

test("the browser timeout exceeds the bounded Hugging Face catalog timeout", () => {
  assert.equal(CATALOG_TIMEOUT_MS, 90_000);
  assert.ok(CATALOG_TIMEOUT_MS > 65_000);
  assert.equal(MODEL_PREFLIGHT_TIMEOUT_MS, 120_000);
});

test("an unchecked gated list entry is not reported as denied", () => {
  assert.equal(
    accessStatus({ gated: true, private: false, access_authorized: false, access_checked: false }),
    "UNVERIFIED",
  );
});

test("only an exact failed access check reports access required", () => {
  assert.equal(
    accessStatus({ gated: true, private: false, access_authorized: false, access_checked: true }),
    "ACCESS_REQUIRED",
  );
  assert.equal(
    accessStatus({ gated: true, private: false, access_authorized: true, access_checked: true }),
    "AUTHORIZED",
  );
});
