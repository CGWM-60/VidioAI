"""Resolution conservative de la precision des pipelines et composants.

La capacite materielle du GPU est un filtre final, jamais une raison suffisante
pour convertir un modele entier. Les metadonnees des poids et de quantification
restent la source de verite prioritaire.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentPrecisionPlan:
    name: str
    dtype: str | None = None
    device: str | None = None
    quantization: str | None = None


@dataclass(slots=True)
class PrecisionPlan:
    load_dtype: str
    precision: str
    source: str
    quantization: str | None = None
    components: list[ComponentPrecisionPlan] = field(default_factory=list)
    recovery_attempted: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            mode="AUTO",
            resolved=self.precision,
            reason=self.source,
        )
        return payload


class DTypeResolver:
    """Construit un plan de precision sans cast global destructeur."""

    _DTYPE_KEYS = {
        "torch_dtype",
        "dtype",
        "weight_dtype",
        "transformer_dtype",
        "unet_dtype",
        "vae_dtype",
        "text_encoder_dtype",
    }
    _COMPONENTS = (
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
    )

    @staticmethod
    def _precision_label(dtype: str) -> str:
        return {
            "float16": "FP16",
            "bfloat16": "BF16",
            "float32": "FP32",
            "auto": "AUTO",
        }[dtype]

    @staticmethod
    def _normalise_dtype(value: Any) -> str | None:
        token = str(value or "").lower().replace("torch.", "").replace("_", "")
        if token in {"bfloat16", "bf16"}:
            return "bfloat16"
        if token in {"float16", "fp16", "half"}:
            return "float16"
        if token in {"float32", "fp32", "float"}:
            return "float32"
        return None

    @classmethod
    def _walk(cls, value: Any, *, path: str = "metadata") -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if str(key).lower() in cls._DTYPE_KEYS:
                    dtype = cls._normalise_dtype(child)
                    if dtype:
                        found.append((child_path, dtype))
                found.extend(cls._walk(child, path=child_path))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                found.extend(cls._walk(child, path=f"{path}[{index}]"))
        return found

    @staticmethod
    def _quantization(metadata: dict[str, Any]) -> str | None:
        config = metadata.get("quantization_config")
        if not isinstance(config, dict):
            for container in (metadata.get("config"), metadata.get("model_index")):
                if isinstance(container, dict) and isinstance(container.get("quantization_config"), dict):
                    config = container["quantization_config"]
                    break
        if not isinstance(config, dict):
            component_configs = metadata.get("component_configs")
            if isinstance(component_configs, dict):
                for component in component_configs.values():
                    if isinstance(component, dict) and isinstance(
                        component.get("quantization_config"), dict
                    ):
                        config = component["quantization_config"]
                        break
        if not isinstance(config, dict):
            return None
        method = config.get("quant_method") or config.get("quantization_method")
        if method:
            return str(method)
        if config.get("load_in_4bit"):
            return "bitsandbytes-4bit"
        if config.get("load_in_8bit"):
            return "bitsandbytes-8bit"
        return "quantized"

    def resolve(self, metadata: dict[str, Any], *, cuda_available: bool, bf16_supported: bool) -> PrecisionPlan:
        quantization = self._quantization(metadata)

        # Un modele quantifie doit laisser Transformers/Diffusers conserver les
        # dtypes heterogenes de ses composants.
        if quantization:
            return PrecisionPlan("auto", "AUTO", "quantization_config", quantization)

        # Dtype global du modele avant toute declaration de composant.
        model_explicit: list[tuple[str, str]] = []
        for name in ("config", "model_index"):
            container = metadata.get(name)
            if isinstance(container, dict):
                for key in self._DTYPE_KEYS:
                    requested = self._normalise_dtype(container.get(key))
                    if requested:
                        model_explicit.append((f"metadata.{name}.{key}", requested))
        if model_explicit:
            source, requested = model_explicit[0]
            if requested == "bfloat16" and (not cuda_available or not bf16_supported):
                requested = "float16" if cuda_available else "float32"
                source = f"gpu_filter({source})"
            elif requested == "float16" and not cuda_available:
                requested = "float32"
                source = f"cpu_filter({source})"
            return PrecisionPlan(requested, self._precision_label(requested), source)

        # Les composants peuvent legitimement etre heterogenes. Dans ce cas le
        # chargeur conserve leurs declarations avec `auto`.
        components: list[ComponentPrecisionPlan] = []
        component_configs = metadata.get("component_configs")
        if isinstance(component_configs, dict):
            for name, config in component_configs.items():
                found = self._walk(config, path=f"component_configs.{name}")
                if found:
                    components.append(ComponentPrecisionPlan(name=name, dtype=found[0][1]))
        if components:
            return PrecisionPlan(
                "auto",
                "AUTO",
                "component_dtype_declarations",
                components=components,
            )

        # Les projections de poids du Hub peuvent publier une liste de dtypes.
        weight_values = metadata.get("tensor_dtypes") or metadata.get("weight_dtypes") or []
        if isinstance(weight_values, dict):
            weight_values = list(weight_values.keys())
        if not isinstance(weight_values, list):
            weight_values = [weight_values]
        weight_dtypes = [self._normalise_dtype(value) for value in weight_values]
        weight_dtypes = [value for value in weight_dtypes if value]
        if weight_dtypes:
            unique = set(weight_dtypes)
            if len(unique) > 1:
                return PrecisionPlan("auto", "AUTO", "mixed_weight_dtypes")
            requested = weight_dtypes[0]
            if requested == "bfloat16" and (not cuda_available or not bf16_supported):
                requested = "float16" if cuda_available else "float32"
            if requested == "float16" and not cuda_available:
                requested = "float32"
            return PrecisionPlan(requested, self._precision_label(requested), "weight_metadata")

        pipeline_dtype = self._normalise_dtype(metadata.get("pipeline_dtype"))
        if pipeline_dtype:
            if pipeline_dtype == "bfloat16" and (not cuda_available or not bf16_supported):
                pipeline_dtype = "float16" if cuda_available else "float32"
            return PrecisionPlan(
                pipeline_dtype,
                self._precision_label(pipeline_dtype),
                "pipeline_constraint",
            )

        # Politique sure : FP16 CUDA, FP32 CPU. BF16 n'est jamais infere de la
        # seule presence d'un GPU compatible.
        dtype = "float16" if cuda_available else "float32"
        return PrecisionPlan(dtype, self._precision_label(dtype), "safe_runtime_policy")

    @staticmethod
    def materialize(torch: Any, plan: PrecisionPlan) -> Any:
        if plan.load_dtype == "auto":
            return "auto"
        return getattr(torch, plan.load_dtype)

    @classmethod
    def component_names(cls) -> tuple[str, ...]:
        return cls._COMPONENTS

    @classmethod
    def inspect_components(cls, pipeline: Any) -> list[ComponentPrecisionPlan]:
        result: list[ComponentPrecisionPlan] = []
        for name in cls._COMPONENTS:
            component = getattr(pipeline, name, None)
            if component is None:
                continue
            dtype = getattr(component, "dtype", None)
            device = getattr(component, "device", None)
            quantization = getattr(component, "quantization_method", None)
            if quantization is None:
                config = getattr(component, "config", None)
                quantization = getattr(config, "quantization_config", None)
            result.append(
                ComponentPrecisionPlan(
                    name=name,
                    dtype=str(dtype).replace("torch.", "") if dtype is not None else None,
                    device=str(device) if device is not None else None,
                    quantization=str(quantization) if quantization is not None else None,
                )
            )
        return result


    @staticmethod
    def is_dtype_mismatch(error: BaseException) -> bool:
        message = f"{type(error).__name__}: {error}".lower()
        markers = (
            "expected scalar type",
            "mat1 and mat2 must have the same dtype",
            "input type (")
        return any(marker in message for marker in markers) and any(
            token in message for token in ("float", "half", "bfloat", "dtype")
        )

    @staticmethod
    def recovery_plan(plan: PrecisionPlan, *, cuda_available: bool) -> PrecisionPlan | None:
        if plan.recovery_attempted:
            return None
        if plan.load_dtype == "bfloat16":
            dtype = "float16" if cuda_available else "float32"
        elif plan.load_dtype == "float16":
            # `auto` laisse Diffusers respecter les dtypes propres aux composants
            # (par exemple VAE FP32 + transformer FP16) sans doubler toute la VRAM.
            dtype = "auto"
        else:
            return None
        return PrecisionPlan(
            load_dtype=dtype,
            precision=DTypeResolver._precision_label(dtype),
            source=f"dtype_mismatch_recovery({plan.load_dtype})",
            recovery_attempted=True,
        )


# Alias de compatibilite interne pour les consommateurs de la premiere version.
ComponentPrecision = ComponentPrecisionPlan
