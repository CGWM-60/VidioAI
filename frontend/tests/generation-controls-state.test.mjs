import assert from "node:assert/strict";
import test from "node:test";

import {
  advancedParameterDescriptors,
  GENERATION_PRESETS,
  generationAdvancedPayload,
  IMAGE_PRESET_FALLBACKS,
} from "../app/lib/generation-controls-state.mjs";

test("simple mode exposes one common preset vocabulary", () => {
  assert.deepEqual(GENERATION_PRESETS, ["FAST", "BALANCED", "QUALITY"]);
  assert.deepEqual(Object.keys(IMAGE_PRESET_FALLBACKS), GENERATION_PRESETS);
});

test("advanced mode exposes only parameters declared by the ModelPack", () => {
  const descriptors = advancedParameterDescriptors({
    model_pack: {
      id: "wan22-i2v",
      supported_parameters: {
        seed: { type: "number", min: 0 },
        scheduler: { options: ["normal", "karras"] },
        strength: false,
      },
    },
  });
  assert.deepEqual(descriptors.map(({ key }) => key), ["seed", "scheduler"]);
  assert.deepEqual(generationAdvancedPayload(descriptors, { seed: "42", scheduler: "karras", strength: 0.5 }), {
    seed: 42,
    scheduler: "karras",
  });
  assert.deepEqual(
    advancedParameterDescriptors({ model_pack: { inputs: { advanced_parameters: ["steps", "resolution"] } } })
      .map(({ key }) => key),
    ["steps", "resolution"],
  );
  assert.deepEqual(advancedParameterDescriptors({}), []);
});
