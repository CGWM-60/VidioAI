"""Resolution UI -> dimensions réellement acceptables par une pipeline vidéo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


QUALITY_SHORT_SIDE = {"480p": 480, "720p": 720, "1080p": 1080}
ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _config_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _scalar(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        values = [_positive_int(item) for item in value]
        values = [item for item in values if item is not None]
        return max(values) if values else None
    return _positive_int(value)


@dataclass(frozen=True, slots=True)
class ResolvedResolution:
    requested_quality: str
    requested_aspect_ratio: str
    width: int
    height: int
    dimension_multiple: int


class ResolutionResolver:
    """Résout une cible sémantique sans supposer que 720p vaut exactement 1280x720."""

    def resolve(
        self,
        *,
        quality: str | None,
        aspect_ratio: str | None,
        pipeline: Any = None,
        metadata: dict[str, Any] | None = None,
        requested_width: Any = None,
        requested_height: Any = None,
        default_width: int = 512,
        default_height: int = 512,
    ) -> ResolvedResolution:
        metadata = metadata or {}
        normalized_quality = str(quality or "").strip().lower()
        normalized_ratio = str(aspect_ratio or "16:9").strip()
        if normalized_quality and normalized_quality not in {*QUALITY_SHORT_SIDE, "auto"}:
            raise ValueError(f"Qualité vidéo non supportée: {normalized_quality}")
        if normalized_ratio not in ASPECT_RATIOS:
            raise ValueError(f"Ratio vidéo non supporté: {normalized_ratio}")

        target_width, target_height = self._target_dimensions(
            normalized_quality,
            normalized_ratio,
            requested_width,
            requested_height,
            default_width,
            default_height,
        )
        multiple = self._dimension_multiple(pipeline, metadata)
        min_width, max_width = self._limits("width", pipeline, metadata)
        min_height, max_height = self._limits("height", pipeline, metadata)
        width, height = self._closest_pair(
            target_width,
            target_height,
            ASPECT_RATIOS[normalized_ratio],
            multiple,
            min_width,
            max_width,
            min_height,
            max_height,
        )
        return ResolvedResolution(
            requested_quality=normalized_quality or "auto",
            requested_aspect_ratio=normalized_ratio,
            width=width,
            height=height,
            dimension_multiple=multiple,
        )

    @staticmethod
    def _target_dimensions(
        quality: str,
        aspect_ratio: str,
        width: Any,
        height: Any,
        default_width: int,
        default_height: int,
    ) -> tuple[float, float]:
        ratio = ASPECT_RATIOS[aspect_ratio]
        short_side = QUALITY_SHORT_SIDE.get(quality)
        if short_side is not None:
            if ratio >= 1:
                return short_side * ratio, float(short_side)
            return float(short_side), short_side / ratio

        explicit_width = _positive_int(width)
        explicit_height = _positive_int(height)
        if explicit_width and explicit_height:
            return float(explicit_width), float(explicit_height)
        if explicit_width:
            return float(explicit_width), explicit_width / ratio
        if explicit_height:
            return explicit_height * ratio, float(explicit_height)
        return float(default_width), float(default_height)

    @staticmethod
    def _metadata_sources(metadata: dict[str, Any]) -> list[Any]:
        model_index = metadata.get("model_index") or {}
        config = metadata.get("config") or {}
        return [
            metadata,
            model_index.get("vidioai_runtime") or {},
            config.get("vidioai_runtime") or {},
            model_index,
            config,
        ]

    def _dimension_multiple(self, pipeline: Any, metadata: dict[str, Any]) -> int:
        sources = self._metadata_sources(metadata)
        explicit = [
            _scalar(_config_value(source, key))
            for source in sources
            for key in ("dimension_multiple", "spatial_multiple", "resolution_multiple")
        ]

        vae = getattr(pipeline, "vae", None)
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        vae_scale = next(
            (
                value
                for value in (
                    _scalar(getattr(pipeline, "vae_scale_factor", None)),
                    _scalar(_config_value(getattr(vae, "config", None), "spatial_compression_ratio")),
                    *(
                        _scalar(_config_value(source, key))
                        for source in sources
                        for key in ("vae_scale_factor", "spatial_compression_ratio")
                    ),
                )
                if value
            ),
            8,
        )
        patch_size = next(
            (
                value
                for value in (
                    _scalar(_config_value(getattr(transformer, "config", None), "patch_size")),
                    _scalar(_config_value(getattr(unet, "config", None), "patch_size")),
                    *(_scalar(_config_value(source, "patch_size")) for source in sources),
                )
                if value
            ),
            1,
        )
        candidates = [value for value in explicit if value]
        candidates.extend([8, vae_scale * patch_size])
        multiple = 1
        for candidate in candidates:
            multiple = math.lcm(multiple, candidate)
        if multiple > 256:
            raise ValueError(f"Multiple spatial du modèle non exploitable: {multiple}")
        return max(8, multiple)

    def _limits(
        self,
        axis: str,
        pipeline: Any,
        metadata: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        sources: list[Any] = self._metadata_sources(metadata)
        for component_name in ("transformer", "unet", "vae"):
            component = getattr(pipeline, component_name, None)
            sources.append(getattr(component, "config", None))
        sources.append(getattr(pipeline, "config", None))

        minimums = [
            _positive_int(_config_value(source, key))
            for source in sources
            for key in (f"min_{axis}", f"minimum_{axis}")
        ]
        maximums = [
            _positive_int(_config_value(source, key))
            for source in sources
            for key in (f"max_{axis}", f"maximum_{axis}")
        ]
        minimum_values = [value for value in minimums if value]
        maximum_values = [value for value in maximums if value]
        return (
            max(minimum_values) if minimum_values else None,
            min(maximum_values) if maximum_values else None,
        )

    @staticmethod
    def _axis_candidates(
        target: float,
        multiple: int,
        minimum: int | None,
        maximum: int | None,
    ) -> list[int]:
        lower = max(multiple, math.ceil((minimum or multiple) / multiple) * multiple)
        upper = math.floor(maximum / multiple) * multiple if maximum else None
        if upper is not None and lower > upper:
            raise ValueError("Les limites de résolution du modèle sont incompatibles avec son multiple spatial.")

        center = round(target / multiple)
        values = {lower}
        for offset in range(-6, 7):
            candidate = max(lower, (center + offset) * multiple)
            if upper is None or candidate <= upper:
                values.add(candidate)
        if upper is not None:
            values.add(upper)
        return sorted(values)

    def _closest_pair(
        self,
        target_width: float,
        target_height: float,
        target_ratio: float,
        multiple: int,
        min_width: int | None,
        max_width: int | None,
        min_height: int | None,
        max_height: int | None,
    ) -> tuple[int, int]:
        widths = self._axis_candidates(target_width, multiple, min_width, max_width)
        heights = self._axis_candidates(target_height, multiple, min_height, max_height)

        def score(pair: tuple[int, int]) -> tuple[float, float]:
            width, height = pair
            ratio_error = abs(math.log((width / height) / target_ratio))
            scale_error = math.hypot(
                (width - target_width) / target_width,
                (height - target_height) / target_height,
            )
            area_error = abs(math.log((width * height) / (target_width * target_height)))
            overshoot = (
                max(0.0, width - target_width) / target_width
                + max(0.0, height - target_height) / target_height
            )
            # La qualité est une cible de taille : une légère erreur de ratio est
            # préférable à un agrandissement global moins fidèle (1280x704 doit
            # ainsi battre 1312x736 pour une cible 720p sur un multiple de 32).
            return (
                ratio_error * 0.75 + scale_error + area_error * 0.25 + overshoot * 0.05,
                scale_error,
            )

        return min(((width, height) for width in widths for height in heights), key=score)
