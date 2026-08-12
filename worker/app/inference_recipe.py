"""Résolution générique des paramètres d'inférence VidioAI.

Priorité:
1. paramètres explicites de la génération;
2. recette installée avec le bundle;
3. defaults concrets de la signature de pipeline pour la résolution QUALITY;
4. sinon omission du kwarg -> comportement natif Diffusers.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any


QUALITY_MODES = {"native", "fast", "balanced", "quality"}


class RecipeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "INFERENCE_RECIPE_INVALID",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class InferenceRecipePlan:
    quality_mode: str
    values: dict[str, Any]
    sources: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceRecipeResolver:
    _RANGES = {
        "width": (64, 2048),
        "height": (64, 2048),
        "num_inference_steps": (1, 100),
        "guidance_scale": (0.0, 30.0),
        "true_cfg_scale": (0.0, 30.0),
        "strength": (0.0, 1.0),
        "max_sequence_length": (16, 8192),
        "inference_fps": (1, 60),
    }

    @staticmethod
    def _signature_default(
        pipeline: Any,
        name: str,
    ) -> Any:
        try:
            parameter = inspect.signature(
                pipeline.__call__
            ).parameters.get(name)
        except (TypeError, ValueError):
            return None
        if (
            parameter is None
            or parameter.default is inspect.Parameter.empty
        ):
            return None
        return parameter.default

    @classmethod
    def normalize_recipe(
        cls,
        raw: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not raw:
            return {}
        if not isinstance(raw, dict):
            raise RecipeError(
                "La recette d'inférence doit être un objet."
            )

        result: dict[str, Any] = {}
        mode = str(
            raw.get("quality_mode") or "native"
        ).strip().lower()
        if mode not in QUALITY_MODES:
            raise RecipeError(
                "quality_mode doit être native, fast, balanced ou quality."
            )
        result["quality_mode"] = mode

        aliases = {
            "steps": "num_inference_steps",
            "num_inference_steps": "num_inference_steps",
            "width": "width",
            "height": "height",
            "guidance_scale": "guidance_scale",
            "true_cfg_scale": "true_cfg_scale",
            "strength": "strength",
            "max_sequence_length": "max_sequence_length",
            "inference_fps": "inference_fps",
        }
        for source, target in aliases.items():
            if source not in raw or raw[source] is None:
                continue
            value = raw[source]
            low, high = cls._RANGES[target]
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise RecipeError(
                    f"{target} doit être numérique."
                ) from error
            if not low <= number <= high:
                raise RecipeError(
                    f"{target} doit être compris entre {low} et {high}."
                )
            if target in {
                "width",
                "height",
                "num_inference_steps",
                "max_sequence_length",
                "inference_fps",
            }:
                value = int(number)
            else:
                value = float(number)
            result[target] = value
        return result

    def resolve(
        self,
        *,
        pipeline: Any,
        request: dict[str, Any],
        bundle: dict[str, Any] | None,
    ) -> InferenceRecipePlan:
        bundle_recipe = (
            bundle.get("recipe")
            if isinstance(bundle, dict)
            else None
        )
        recipe = self.normalize_recipe(
            bundle_recipe
            if isinstance(bundle_recipe, dict)
            else {}
        )

        requested_mode = str(
            request.get("quality") or ""
        ).strip().lower()
        if requested_mode not in QUALITY_MODES:
            requested_mode = ""
        quality_mode = (
            requested_mode
            or str(recipe.get("quality_mode") or "native")
        )

        values: dict[str, Any] = {}
        sources: dict[str, str] = {}

        request_aliases = {
            "width": ("width",),
            "height": ("height",),
            "num_inference_steps": (
                "steps",
                "num_inference_steps",
            ),
            "guidance_scale": ("guidance_scale",),
            "true_cfg_scale": ("true_cfg_scale",),
            "strength": ("strength",),
            "max_sequence_length": (
                "max_sequence_length",
            ),
            "inference_fps": ("inference_fps",),
        }

        for canonical, aliases in request_aliases.items():
            explicit = None
            for alias in aliases:
                if (
                    alias in request
                    and request.get(alias) is not None
                ):
                    explicit = request.get(alias)
                    break
            if explicit is not None:
                checked = self.normalize_recipe(
                    {
                        "quality_mode": quality_mode,
                        canonical: explicit,
                    }
                )
                values[canonical] = checked[canonical]
                sources[canonical] = "request"
                continue

            if canonical in recipe:
                values[canonical] = recipe[canonical]
                sources[canonical] = "bundle_recipe"
                continue

            # QUALITY ne devine jamais une résolution. Il matérialise seulement
            # un default numérique que la vraie signature Diffusers annonce.
            if (
                quality_mode == "quality"
                and canonical in {"width", "height"}
            ):
                native = self._signature_default(
                    pipeline,
                    canonical,
                )
                if (
                    isinstance(native, int)
                    and not isinstance(native, bool)
                    and 64 <= native <= 2048
                ):
                    values[canonical] = native
                    sources[canonical] = "pipeline_signature"

        return InferenceRecipePlan(
            quality_mode=quality_mode,
            values=values,
            sources=sources,
        )
