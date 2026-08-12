"""Adapter générique Diffusers ModularPipeline.

Le chargeur s'appuie uniquement sur ``modular_model_index.json`` et l'API
publique de Diffusers. Aucun repository ou nom MiniMax/Wan/etc. n'est codé en
dur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..capability_resolver import CapabilityResolver
from ..modular_runtime import ModularManifestResolver, ModularRuntimeError
from ..normalizers import NormalizationError
from .base import RuntimeAdapter, log_diffusers_call


_VIDEO_CAPABILITIES = {
    "TEXT_TO_VIDEO",
    "IMAGE_TO_VIDEO",
    "MULTI_IMAGE_TO_VIDEO",
    "START_END_IMAGE_TO_VIDEO",
    "KEYFRAMES_TO_VIDEO",
    "VIDEO_TO_VIDEO",
    "VIDEO_INPAINTING",
    "VIDEO_UPSCALE",
}


class ModularDiffusersAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return [
            "TEXT_TO_IMAGE",
            "IMAGE_TO_IMAGE",
            "INPAINTING",
            "OUTPAINTING",
            "IMAGE_VARIATION",
            "IMAGE_UPSCALE",
            "CONTROLLED_IMAGE_GENERATION",
            "TEXT_TO_VIDEO",
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
            "VIDEO_TO_VIDEO",
            "VIDEO_INPAINTING",
            "VIDEO_UPSCALE",
        ]

    def supported_capabilities(
        self,
        metadata: dict[str, Any],
    ) -> list[str]:
        # Après le premier chargement, le RuntimeManager écrit les capacités
        # réellement observées dans les blocks. Elles priment sur les tags HF.
        observed = metadata.get("runtime_capabilities") or metadata.get("capabilities") or []
        observed_set = {
            str(value).strip().upper()
            for value in observed
            if isinstance(value, str) and value.strip()
        }
        if observed_set:
            return [
                value
                for value in self.capabilities()
                if value in observed_set
            ]

        declared = CapabilityResolver().declared_capabilities(metadata)
        return [
            value for value in self.capabilities() if value in declared
        ]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        return ModularManifestResolver.is_modular(metadata)

    def estimate_resources(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        del metadata
        # L'autorité réelle reste MemoryPlanner après téléchargement des
        # composants. Une estimation fixe ici serait trompeuse.
        return {
            "vram_bytes": 0,
            "ram_bytes": 0,
        }

    @staticmethod
    def _load_kwargs(
        dtype: Any,
        *,
        local_files_only: bool = True,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if dtype not in {None, "auto"}:
            result["torch_dtype"] = dtype
        return result

    def load(
        self,
        snapshot: str,
        settings: dict[str, Any],
        runtime: Any,
    ) -> Any:
        try:
            from diffusers import ComponentsManager, ModularPipeline
        except (ImportError, ModuleNotFoundError) as error:
            raise ModularRuntimeError(
                "La version Diffusers installée ne fournit pas ModularPipeline.",
                code="MODULAR_PIPELINE_UNAVAILABLE",
                status_code=503,
            ) from error

        metadata = runtime.get("metadata") or {}
        modular_index = ModularManifestResolver.from_metadata(metadata)
        config = metadata.get("config")
        if ModularManifestResolver.requires_remote_code(
            modular_index,
            config if isinstance(config, dict) else {},
        ):
            raise ModularRuntimeError(
                "Ce ModularPipeline exige du code distant; trust_remote_code reste refusé.",
                code="REMOTE_CODE_REQUIRED",
            )

        manager = ComponentsManager()
        pipeline = ModularPipeline.from_pretrained(
            snapshot,
            components_manager=manager,
            trust_remote_code=False,
            local_files_only=True,
        )

        dtype = settings.get("torch_dtype")
        memory_plan = runtime.get("memory_plan")
        strategy = getattr(memory_plan, "strategy", None)
        device = str(settings.get("device") or "cpu")

        # Les dépendances externes ont été copiées dans le snapshot pendant
        # l'installation. On les charge composant par composant avec un chemin
        # local explicite ; aucune résolution réseau n'est nécessaire.
        for record in ModularManifestResolver.read_materialization(snapshot):
            name = str(record.get("name") or "")
            local_root = record.get("local_root")
            if not name or not isinstance(local_root, str):
                continue
            component_root = Path(snapshot) / local_root
            if not component_root.is_dir():
                raise ModularRuntimeError(
                    f"Composant modular local absent: {name}",
                    code="MODULAR_COMPONENT_MISSING",
                )
            kwargs = self._load_kwargs(dtype)
            kwargs["pretrained_model_name_or_path"] = str(component_root)
            # Le chemin est local : neutralise la révision distante conservée
            # dans le ComponentSpec afin de ne déclencher aucune résolution Hub.
            kwargs["revision"] = None
            pipeline.load_components(
                names=[name],
                **kwargs,
            )

        remaining_kwargs = self._load_kwargs(dtype)

        # Les composants déjà chargés ci-dessus sont ignorés par Diffusers.
        pipeline.load_components(**remaining_kwargs)

        managed = False
        if device == "cuda" and strategy == "FULL_GPU":
            # C'est le chemin documenté par Diffusers pour ModularPipeline.
            pipeline.to("cuda")
        elif device == "cuda" and strategy in {
            "MODEL_CPU_OFFLOAD",
            "SEQUENTIAL_CPU_OFFLOAD",
        }:
            manager.enable_auto_cpu_offload(
                device="cuda:0",
                memory_reserve_margin="3GB",
            )
            managed = True

        setattr(pipeline, "_vidioai_modular", True)
        setattr(
            pipeline,
            "_vidioai_modular_full_gpu",
            device == "cuda" and strategy == "FULL_GPU",
        )
        setattr(
            pipeline,
            "_vidioai_components_manager_managed",
            managed,
        )
        setattr(
            pipeline,
            "_vidioai_modular_component_names",
            [
                spec.name
                for spec in ModularManifestResolver.components(modular_index)
            ],
        )
        setattr(
            pipeline,
            "_vidioai_components_manager",
            manager,
        )
        return pipeline

    def unload(
        self,
        pipeline: Any,
        runtime: Any,
    ) -> None:
        del runtime
        manager = getattr(
            pipeline,
            "_vidioai_components_manager",
            None,
        )
        if manager is not None:
            disable = getattr(
                manager,
                "disable_auto_cpu_offload",
                None,
            )
            if callable(disable):
                try:
                    disable()
                except Exception:
                    pass
        del pipeline

    @staticmethod
    def _block_input_names(pipeline: Any) -> set[str]:
        blocks = getattr(pipeline, "blocks", None)
        values: set[str] = set()
        for item in getattr(blocks, "inputs", []) or []:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name:
                values.add(name)
            elif isinstance(item, str) and item:
                values.add(item)
        return values

    @staticmethod
    def _assign_first(
        kwargs: dict[str, Any],
        accepted: set[str],
        value: Any,
        *aliases: str,
    ) -> None:
        if value is None:
            return
        for name in aliases:
            if not accepted or name in accepted:
                kwargs[name] = value
                return

    @staticmethod
    def _resolved_images_and_roles(
        request: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        images = list(request.get("resolved_input_images") or [])
        raw = sorted(
            [
                item
                for item in request.get("input_images") or []
                if isinstance(item, dict)
            ],
            key=lambda item: int(item.get("order") or 0),
        )
        roles = [
            str(item.get("role") or "reference").lower()
            for item in raw
        ]
        if len(roles) < len(images):
            roles.extend(["reference"] * (len(images) - len(roles)))
        return images, roles[: len(images)]

    @classmethod
    def _payload_dict(cls, output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            return dict(output)

        to_dict = getattr(output, "to_dict", None)
        if callable(to_dict):
            try:
                value = to_dict()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass

        values = getattr(output, "values", None)
        if isinstance(values, dict):
            return dict(values)

        result: dict[str, Any] = {}
        for name in (
            "images",
            "image",
            "frames",
            "video",
            "videos",
            "audio",
            "audios",
            "sample_rate",
            "audio_sample_rate",
        ):
            value = getattr(output, name, None)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def _normalize_output(
        cls,
        output: Any,
        *,
        video: bool,
    ) -> dict[str, Any]:
        payload = cls._payload_dict(output)

        if video:
            for name in ("frames", "video", "videos", "images"):
                value = payload.get(name)
                if value is not None:
                    result = {"frames": value}
                    # L'audio est conservé dans le contrat interne, mais le
                    # muxeur VidioAI n'est pas encore activé. On ne prétend donc
                    # pas que l'audio est livré dans le MP4.
                    for audio_name in ("audio", "audios"):
                        if payload.get(audio_name) is not None:
                            result["native_audio"] = payload[audio_name]
                            break
                    if payload.get("sample_rate") is not None:
                        result["audio_sample_rate"] = payload["sample_rate"]
                    elif payload.get("audio_sample_rate") is not None:
                        result["audio_sample_rate"] = payload[
                            "audio_sample_rate"
                        ]
                    return result
        else:
            for name in ("images", "image"):
                value = payload.get(name)
                if value is not None:
                    return {
                        "images": value
                        if isinstance(value, (list, tuple))
                        else [value]
                    }

        if isinstance(output, (list, tuple)) and output:
            return {
                "frames" if video else "images": output
            }

        raise NormalizationError(
            "Le ModularPipeline n'a produit aucune sortie média exploitable.",
            code="OUTPUT_NORMALIZATION_FAILED",
        )

    def generate(
        self,
        pipeline: Any,
        runtime: Any,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        capability = str(
            request.get("capability") or "TEXT_TO_VIDEO"
        ).upper()
        video = capability in _VIDEO_CAPABILITIES
        accepted = self._block_input_names(pipeline)

        kwargs: dict[str, Any] = {}
        self._assign_first(
            kwargs,
            accepted,
            request.get("prompt"),
            "prompt",
        )
        self._assign_first(
            kwargs,
            accepted,
            request.get("negative_prompt"),
            "negative_prompt",
        )

        scalar_aliases = [
            ("width", request.get("width")),
            ("height", request.get("height")),
            ("num_inference_steps", request.get("steps")),
            ("guidance_scale", request.get("guidance_scale")),
            ("true_cfg_scale", request.get("true_cfg_scale")),
            ("max_sequence_length", request.get("max_sequence_length")),
            ("fps", request.get("inference_fps")),
        ]
        for name, value in scalar_aliases:
            self._assign_first(kwargs, accepted, value, name)

        frames = request.get("frames")
        self._assign_first(
            kwargs,
            accepted,
            frames,
            "num_frames",
            "video_length",
        )

        generator = runtime.get("generator")
        self._assign_first(
            kwargs,
            accepted,
            generator,
            "generator",
        )

        images, roles = self._resolved_images_and_roles(request)
        if request.get("input_image") is not None and not images:
            images = [request["input_image"]]
            roles = ["start_frame"]

        start_image = None
        end_image = None
        references: list[Any] = []
        for index, image in enumerate(images):
            role = roles[index] if index < len(roles) else "reference"
            if role in {"start", "start_frame", "first", "first_frame"}:
                start_image = start_image or image
            elif role in {"end", "end_frame", "last", "last_frame"}:
                end_image = end_image or image
            else:
                references.append(image)

        if start_image is None and images:
            start_image = images[0]
        if (
            end_image is None
            and capability == "START_END_IMAGE_TO_VIDEO"
            and len(images) > 1
        ):
            end_image = images[-1]

        if capability in {
            "IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        }:
            self._assign_first(
                kwargs,
                accepted,
                start_image,
                "image",
                "start_image",
                "first_image",
                "first_frame",
            )

        if capability == "START_END_IMAGE_TO_VIDEO":
            self._assign_first(
                kwargs,
                accepted,
                end_image,
                "last_image",
                "end_image",
                "last_frame",
            )

        if capability in {
            "MULTI_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        }:
            values = references or images
            self._assign_first(
                kwargs,
                accepted,
                values,
                "reference_images",
                "images",
                "conditioning_images",
                "keyframes",
            )

        input_video = request.get("input_video")
        if input_video is None:
            input_video = request.get("input_frames")
        if capability in {
            "VIDEO_TO_VIDEO",
            "VIDEO_INPAINTING",
            "VIDEO_UPSCALE",
        }:
            self._assign_first(
                kwargs,
                accepted,
                input_video,
                "video",
                "frames",
                "reference_video",
            )

        if request.get("mask_image") is not None:
            self._assign_first(
                kwargs,
                accepted,
                request.get("mask_image"),
                "mask_image",
                "mask",
            )

        # Une pipeline modular sans propriété ``blocks.inputs`` est considérée
        # opaque : passer des kwargs au hasard serait une fausse compatibilité.
        if not accepted:
            raise ModularRuntimeError(
                "Les entrées du ModularPipeline ne sont pas introspectables.",
                code="MODULAR_INPUT_CONTRACT_UNKNOWN",
            )

        log_diffusers_call(pipeline, kwargs, capability)
        output = pipeline(**kwargs)
        return self._normalize_output(output, video=video)

    def input_profile(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        declared = set(
            CapabilityResolver().declared_capabilities(metadata or {})
        )
        multi = bool(
            declared
            & {
                "MULTI_IMAGE_TO_VIDEO",
                "KEYFRAMES_TO_VIDEO",
                "START_END_IMAGE_TO_VIDEO",
            }
        )
        return {
            "min_input_images": 0
            if "TEXT_TO_VIDEO" in declared
            else 1,
            "max_input_images": 8 if multi else 2,
            "supported_image_roles": [
                "start_frame",
                "end_frame",
                "reference",
                "keyframe",
            ],
            "supports_start_end_frames": (
                "START_END_IMAGE_TO_VIDEO" in declared
            ),
            "supports_reference_images": (
                "MULTI_IMAGE_TO_VIDEO" in declared
            ),
            "supports_keyframes": (
                "KEYFRAMES_TO_VIDEO" in declared
            ),
        }
