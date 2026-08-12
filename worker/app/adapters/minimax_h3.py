"""Runtime d'architecture MiniMax-H3 pour Diffusers ModularPipeline.

Le choix se fait sur la classe/architecture publiée par Diffusers
(`MiniMaxH3ModularPipeline`), jamais sur un identifiant de repository.

Source du contrat runtime: intégration MiniMax-H3 de Diffusers 0.40.0.dev0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..modular_runtime import ModularRuntimeError
from .base import RuntimeAdapter, log_diffusers_call


GIB = 1024**3
H3_ARCHITECTURE = "MiniMaxH3ModularPipeline"
H3_FPS = 24
H3_MIN_DURATION = 5.0
H3_MAX_DURATION = 15.0
H3_HOST_RAM_INT8_BYTES = 75 * GIB


class MiniMaxH3Adapter(RuntimeAdapter):
    """Adapter architecture-specific, sans branchement par repo Hugging Face."""

    def capabilities(self) -> list[str]:
        return [
            "TEXT_TO_VIDEO",
            "IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
        ]

    @staticmethod
    def _architecture_tokens(metadata: dict[str, Any]) -> set[str]:
        values: list[Any] = [
            metadata.get("class_name"),
            *(metadata.get("architectures") or []),
        ]
        modular = metadata.get("modular_model_index")
        if isinstance(modular, dict):
            values.extend(
                [
                    modular.get("_class_name"),
                    modular.get("_blocks_class_name"),
                ]
            )
        return {
            str(value).strip()
            for value in values
            if isinstance(value, str) and value.strip()
        }

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        tokens = self._architecture_tokens(metadata)
        if H3_ARCHITECTURE in tokens:
            return True
        # La classe des blocks est également un contrat d'architecture natif
        # Diffusers. Ceci permet l'analyse d'un manifest qui déclare les blocks
        # plutôt que la classe pipeline.
        return any(
            value.startswith("MiniMaxH3")
            for value in tokens
        )

    def supported_capabilities(
        self,
        metadata: dict[str, Any],
    ) -> list[str]:
        del metadata
        return self.capabilities()

    def estimate_resources(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        del metadata
        # Les tailles réelles sont calculées sur les sous-dossiers du snapshot;
        # ne pas inventer un chiffre depuis le nom du modèle.
        return {
            "vram_bytes": 0,
            "ram_bytes": 0,
        }

    @staticmethod
    def workflow_for_capability(capability: str) -> str:
        capability = capability.upper()
        if capability == "TEXT_TO_VIDEO":
            return "t2va"
        if capability in {
            "IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
        }:
            return "fl2va"
        if capability == "MULTI_IMAGE_TO_VIDEO":
            return "ref2va"
        raise ModularRuntimeError(
            f"Capability MiniMax-H3 non mappée: {capability}",
            code="H3_CAPABILITY_UNSUPPORTED",
        )

    @staticmethod
    def _dir_bytes(path: Path) -> int:
        if not path.is_dir():
            return 0
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )

    @classmethod
    def _working_set_bytes(
        cls,
        snapshot: Path,
        workflow: str,
    ) -> int:
        transformer = (
            "transformer_ref"
            if workflow == "ref2va"
            else "transformer"
        )
        # Seuls les sous-dossiers utiles au workflow sont comptés. H3 contient
        # deux partitions transformer dans un même repo et il serait faux de
        # sommer les deux pour t2va/fl2va.
        names = [
            transformer,
            "text_encoder",
            "vae",
            "audio_vae",
            "scheduler",
            "audio_scheduler",
        ]
        return sum(cls._dir_bytes(snapshot / name) for name in names)

    @staticmethod
    def _memory_numbers(memory_plan: Any) -> tuple[int, int]:
        return (
            int(getattr(memory_plan, "vram_total_bytes", 0) or 0),
            int(getattr(memory_plan, "ram_available_bytes", 0) or 0),
        )

    @classmethod
    def _should_use_int8(
        cls,
        snapshot: str | Path,
        workflow: str,
        vram_total_bytes: int,
    ) -> bool:
        if vram_total_bytes <= 0:
            return False
        root = Path(snapshot)
        transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"
        transformer = cls._dir_bytes(root / transformer_name)
        text_encoder = cls._dir_bytes(root / "text_encoder")
        largest_component = max(transformer, text_encoder)
        if largest_component <= 0:
            # Sans taille mesurée, la politique prudente est INT8 plutôt que
            # prétendre que le composant BF16 tiendra sur l'accélérateur.
            return True
        # La recette BF16 officielle réserve 12 GB autour du composant actif.
        return largest_component + 12 * GIB > vram_total_bytes

    @classmethod
    def planning_model_bytes(
        cls,
        snapshot: str | Path,
        capability: str,
        *,
        vram_total_bytes: int = 0,
        ram_available_bytes: int = 0,
    ) -> int:
        del ram_available_bytes
        root = Path(snapshot)
        workflow = cls.workflow_for_capability(capability)
        transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"
        transformer = cls._dir_bytes(root / transformer_name)
        text_encoder = cls._dir_bytes(root / "text_encoder")
        shared = sum(
            cls._dir_bytes(root / name)
            for name in ("vae", "audio_vae", "scheduler", "audio_scheduler")
        )
        bf16 = transformer + text_encoder + shared
        if cls._should_use_int8(root, workflow, vram_total_bytes):
            # La recette officielle quantifie précisément les deux gros composants
            # de BF16 (~2 bytes/weight) vers INT8 (~1 byte/weight).
            return transformer // 2 + text_encoder // 2 + shared
        return bf16

    @staticmethod
    def _require_h3_diffusers() -> dict[str, Any]:
        try:
            import torch
            from diffusers import (
                ComponentsManager,
                MiniMaxH3ModularPipeline,
                MiniMaxH3Transformer3DModel,
                ModularPipeline,
                TorchAoConfig,
            )
            from diffusers.hooks import apply_group_offloading
            from diffusers.modular_pipelines.minimax_h3 import (
                MiniMaxH3AudioReference,
                MiniMaxH3ImageReference,
                MiniMaxH3VideoReference,
            )
            from torchao.quantization import Int8WeightOnlyConfig
            from transformers import (
                Qwen3VLForConditionalGeneration,
                TorchAoConfig as TransformersTorchAoConfig,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise ModularRuntimeError(
                "Le runtime Diffusers MiniMax-H3 n'est pas présent dans "
                f"l'image Worker: {type(error).__name__}: {error}",
                code="H3_RUNTIME_DEPENDENCY_MISSING",
                status_code=503,
            ) from error

        return {
            "torch": torch,
            "ComponentsManager": ComponentsManager,
            "MiniMaxH3ModularPipeline": MiniMaxH3ModularPipeline,
            "MiniMaxH3Transformer3DModel": MiniMaxH3Transformer3DModel,
            "ModularPipeline": ModularPipeline,
            "TorchAoConfig": TorchAoConfig,
            "apply_group_offloading": apply_group_offloading,
            "MiniMaxH3AudioReference": MiniMaxH3AudioReference,
            "MiniMaxH3ImageReference": MiniMaxH3ImageReference,
            "MiniMaxH3VideoReference": MiniMaxH3VideoReference,
            "Int8WeightOnlyConfig": Int8WeightOnlyConfig,
            "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
            "TransformersTorchAoConfig": TransformersTorchAoConfig,
        }

    @classmethod
    def _load_int8(
        cls,
        *,
        pipeline: Any,
        snapshot: Path,
        workflow: str,
        modules: dict[str, Any],
        vram_total: int,
    ) -> None:
        torch = modules["torch"]
        transformer_cls = modules["MiniMaxH3Transformer3DModel"]
        torchao_config = modules["TorchAoConfig"]
        int8 = modules["Int8WeightOnlyConfig"]
        qwen_cls = modules["Qwen3VLForConditionalGeneration"]
        transformers_torchao = modules["TransformersTorchAoConfig"]
        apply_group_offloading = modules["apply_group_offloading"]

        transformer_name = (
            "transformer_ref"
            if workflow == "ref2va"
            else "transformer"
        )

        transformer = transformer_cls.from_pretrained(
            str(snapshot),
            subfolder=transformer_name,
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=torchao_config(
                int8(version=2),
                modules_to_not_convert=[
                    "proj_in",
                    "audio_proj_in",
                    "context_embedder",
                    "time_embedder",
                    "time_proj",
                    "token_refiner",
                    "norm_out",
                    "proj_out",
                    "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=False,
        )
        text_encoder = qwen_cls.from_pretrained(
            str(snapshot),
            subfolder="text_encoder",
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=transformers_torchao(
                int8(version=2),
                modules_to_not_convert=[
                    "model.visual",
                    "model.language_model.embed_tokens",
                    "model.language_model.norm",
                    "lm_head",
                ],
            ),
        )

        pipeline.update_components(
            **{
                transformer_name: transformer,
                "text_encoder": text_encoder,
            }
        )
        pipeline.load_components(
            workflow=workflow,
            dtype=torch.bfloat16,
            local_files_only=True,
        )

        transformer.requires_grad_(False)
        text_encoder.requires_grad_(False)

        offload = {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "use_stream": True,
        }
        transformer.enable_group_offload(
            offload_type="block_level",
            num_blocks_per_group=1,
            **offload,
        )
        apply_group_offloading(
            text_encoder.model,
            offload_type="leaf_level",
            **offload,
        )

        # Recette Diffusers officielle: les VAE restent CUDA sur 24–32 Go.
        # Pour 12–16 Go, le VAE vidéo est lui aussi offloadé.
        if vram_total <= 16 * GIB:
            apply_group_offloading(
                pipeline.vae,
                offload_type="leaf_level",
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                use_stream=False,
            )
        else:
            pipeline.vae.to("cuda")
        pipeline.audio_vae.to("cuda")

        setattr(pipeline, "_vidioai_h3_precision_mode", "INT8")
        setattr(pipeline, "_vidioai_components_manager_managed", True)

    def load(
        self,
        snapshot: str,
        settings: dict[str, Any],
        runtime: Any,
    ) -> Any:
        modules = self._require_h3_diffusers()
        torch = modules["torch"]
        ComponentsManager = modules["ComponentsManager"]
        ModularPipeline = modules["ModularPipeline"]

        capability = str(
            runtime.get("capability") or "TEXT_TO_VIDEO"
        ).upper()
        workflow = self.workflow_for_capability(capability)
        memory_plan = runtime.get("memory_plan")
        vram_total, ram_available = self._memory_numbers(
            memory_plan
        )
        device = str(settings.get("device") or "cpu")
        if device != "cuda":
            raise ModularRuntimeError(
                "MiniMax-H3 nécessite le runtime CUDA VidioAI.",
                code="H3_CUDA_REQUIRED",
                status_code=409,
            )
        if vram_total and vram_total < 12 * GIB:
            raise ModularRuntimeError(
                "Aucune recette locale officielle MiniMax-H3 n'est validée "
                "par VidioAI sous 12 Go de VRAM.",
                code="H3_HARDWARE_BELOW_OFFICIAL_RECIPE",
                status_code=409,
            )

        manager = ComponentsManager()
        pipeline = ModularPipeline.from_pretrained(
            snapshot,
            workflow=workflow,
            components_manager=manager,
            trust_remote_code=False,
            local_files_only=True,
        )

        # Recettes publiées par l'intégration Diffusers:
        # - grosse VRAM : BF16 + ComponentsManager auto-offload;
        # - GPU grand public : INT8 TorchAO + group offload;
        # le choix dépend uniquement des ressources, pas du nom du GPU.
        use_int8 = self._should_use_int8(
            snapshot,
            workflow,
            vram_total,
        )

        if use_int8:
            if ram_available and ram_available < H3_HOST_RAM_INT8_BYTES:
                raise ModularRuntimeError(
                    "La recette INT8 MiniMax-H3 requiert environ 75 Go de RAM "
                    "hôte disponible pour conserver les poids/offloads.",
                    code="H3_INSUFFICIENT_HOST_RAM",
                    status_code=409,
                )
            self._load_int8(
                pipeline=pipeline,
                snapshot=Path(snapshot),
                workflow=workflow,
                modules=modules,
                vram_total=vram_total,
            )
        else:
            dtype = settings.get("torch_dtype")
            if dtype in {None, "auto"}:
                dtype = torch.bfloat16
            pipeline.load_components(
                workflow=workflow,
                dtype=dtype,
                local_files_only=True,
            )
            # Même sur une carte 80 Go, transformer + Qwen3-VL ne tiennent pas
            # ensemble: Diffusers recommande ComponentsManager auto-offload.
            manager.enable_auto_cpu_offload(
                device="cuda",
                memory_reserve_margin="12GB",
            )
            setattr(
                pipeline,
                "_vidioai_components_manager_managed",
                True,
            )
            setattr(
                pipeline,
                "_vidioai_h3_precision_mode",
                "BF16_AUTO_OFFLOAD",
            )

        setattr(pipeline, "_vidioai_modular", True)
        setattr(pipeline, "_vidioai_h3", True)
        setattr(pipeline, "_vidioai_h3_workflow", workflow)
        setattr(
            pipeline,
            "_vidioai_modular_component_names",
            [
                "transformer_ref"
                if workflow == "ref2va"
                else "transformer",
                "text_encoder",
                "vae",
                "audio_vae",
                "scheduler",
                "audio_scheduler",
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
    def _aligned_frames(duration: float | None) -> int | None:
        if duration is None:
            return None
        if duration < H3_MIN_DURATION or duration > H3_MAX_DURATION:
            raise ModularRuntimeError(
                f"MiniMax-H3 accepte 5 à 15 secondes, reçu {duration:g}s.",
                code="H3_DURATION_UNSUPPORTED",
                status_code=422,
            )
        frames = max(1, int(round(duration * H3_FPS)))
        # Contrat VAE H3: num_frames = 17*n + 5, arrondi vers le haut.
        while frames % 17 != 5:
            frames += 1
        if frames / H3_FPS > H3_MAX_DURATION:
            raise ModularRuntimeError(
                "L'alignement VAE dépasserait la durée maximale H3.",
                code="H3_DURATION_UNSUPPORTED",
                status_code=422,
            )
        return frames

    @staticmethod
    def _canvas(value: Any) -> int | None:
        if value is None:
            return None
        parsed = max(32, int(value))
        return max(32, round(parsed / 32) * 32)

    @staticmethod
    def _images_and_roles(
        request: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        images = list(
            request.get("resolved_input_images") or []
        )
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
    def _build_references(
        cls,
        request: dict[str, Any],
        modules: dict[str, Any],
    ) -> list[Any]:
        image_reference = modules["MiniMaxH3ImageReference"]
        video_reference = modules["MiniMaxH3VideoReference"]
        audio_reference = modules["MiniMaxH3AudioReference"]

        ordered: list[tuple[int, Any]] = []
        images, _roles = cls._images_and_roles(request)
        for index, image in enumerate(images):
            ordered.append(
                (index, image_reference(image=image))
            )

        base_order = len(ordered)
        for offset, path in enumerate(
            request.get("reference_video_paths") or []
        ):
            ordered.append(
                (
                    base_order + offset,
                    video_reference.from_file(path),
                )
            )
        base_order = len(ordered)
        for offset, path in enumerate(
            request.get("reference_audio_paths") or []
        ):
            ordered.append(
                (
                    base_order + offset,
                    audio_reference.from_file(path),
                )
            )

        references = [
            value
            for _order, value in sorted(
                ordered,
                key=lambda item: item[0],
            )
        ]
        image_count = sum(
            getattr(item, "kind", "") == "image"
            for item in references
        )
        video_count = sum(
            getattr(item, "kind", "") == "video"
            for item in references
        )
        audio_count = sum(
            getattr(item, "kind", "") == "audio"
            for item in references
        )
        if image_count > 9 or video_count > 3 or audio_count > 3:
            raise ModularRuntimeError(
                "Ref2VA dépasse les limites: 9 images, 3 vidéos, 3 audios.",
                code="H3_REFERENCE_LIMIT_EXCEEDED",
                status_code=422,
            )
        if len(references) > 12:
            raise ModularRuntimeError(
                "Ref2VA accepte au maximum 12 références.",
                code="H3_REFERENCE_LIMIT_EXCEEDED",
                status_code=422,
            )
        if audio_count and image_count + video_count == 0:
            raise ModularRuntimeError(
                "Une référence audio H3 doit être accompagnée d'au moins "
                "une référence image ou vidéo.",
                code="H3_AUDIO_REFERENCE_REQUIRES_VISUAL",
                status_code=422,
            )
        return references

    @staticmethod
    def _normalize_output(output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            payload = output
        else:
            values = getattr(output, "values", None)
            payload = values if isinstance(values, dict) else {}

        videos = (
            payload.get("videos")
            or payload.get("frames")
            or getattr(output, "videos", None)
            or getattr(output, "frames", None)
        )
        audio = (
            payload.get("audio")
            if payload.get("audio") is not None
            else getattr(output, "audio", None)
        )
        rate = (
            payload.get("sampling_rate")
            or payload.get("sample_rate")
            or getattr(output, "sampling_rate", None)
            or getattr(output, "sample_rate", None)
        )
        if videos is None:
            raise ModularRuntimeError(
                "MiniMax-H3 n'a renvoyé aucune vidéo.",
                code="H3_VIDEO_OUTPUT_MISSING",
                status_code=500,
            )
        result = {"frames": videos}
        if audio is not None:
            result["native_audio"] = audio
        if rate is not None:
            result["audio_sample_rate"] = rate
        return result

    def generate(
        self,
        pipeline: Any,
        runtime: Any,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        modules = self._require_h3_diffusers()
        capability = str(
            request.get("capability") or "TEXT_TO_VIDEO"
        ).upper()
        workflow = self.workflow_for_capability(capability)

        duration = (
            request.get("requested_duration_seconds")
            if request.get("requested_duration_seconds") is not None
            else request.get("duration_seconds")
        )
        duration = float(duration) if duration is not None else None
        num_frames = self._aligned_frames(duration)

        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "generator": runtime.get("generator"),
            # H3 génère audio+vidéo ensemble. Demander explicitement les trois
            # sorties évite de dépendre du state complet de ModularPipeline.
            "output": ["videos", "audio", "sampling_rate"],
        }
        if num_frames is not None:
            kwargs["num_frames"] = num_frames

        width = self._canvas(request.get("width"))
        height = self._canvas(request.get("height"))
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height

        # H3 est guidance-distilled: jamais de negative_prompt/guidance_scale.
        if request.get("steps") is not None:
            kwargs["num_inference_steps"] = int(request["steps"])

        images, roles = self._images_and_roles(request)
        if workflow == "fl2va":
            first = None
            last = None
            for index, image in enumerate(images):
                role = roles[index] if index < len(roles) else "start_frame"
                if role in {"end", "end_frame", "last", "last_frame"}:
                    last = last or image
                else:
                    first = first or image
            if first is None and images:
                first = images[0]
            if (
                last is None
                and capability == "START_END_IMAGE_TO_VIDEO"
                and len(images) > 1
            ):
                last = images[-1]
            if first is not None:
                kwargs["image"] = first
            if last is not None:
                kwargs["last_image"] = last

        if workflow == "ref2va":
            references = self._build_references(
                request,
                modules,
            )
            if not references:
                raise ModularRuntimeError(
                    "Ref2VA requiert au moins une référence.",
                    code="H3_REFERENCE_REQUIRED",
                    status_code=422,
                )
            kwargs["references"] = references

        log_diffusers_call(
            pipeline,
            kwargs,
            capability,
        )
        output = pipeline(**kwargs)
        return self._normalize_output(output)

    def input_profile(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del metadata
        return {
            "min_input_images": 0,
            "max_input_images": 9,
            "supported_image_roles": [
                "start_frame",
                "end_frame",
                "reference",
            ],
            "supports_start_end_frames": True,
            "supports_reference_images": True,
            "supports_keyframes": False,
            "supports_reference_videos": True,
            "supports_reference_audio": True,
            "supports_native_audio_output": True,
            "reference_video_limit": 3,
            "reference_audio_limit": 3,
            "reference_total_limit": 12,
        }
