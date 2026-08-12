"""Plan de livraison temporelle distinct des frames d'inference modele."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TemporalOutputPlan:
    native_frames: int
    requested_duration_seconds: float | None
    delivery_fps: int
    delivery_frames: int
    target_duration_seconds: float
    source_fps_for_motion_interpolation: float | None
    strategy: str
    tolerance_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalOutputPlanner:
    """Transforme N frames natives en un fichier respectant duree + FPS produit."""

    def plan(
        self,
        *,
        native_frames: int,
        requested_duration_seconds: float | int | None,
        requested_fps: int | None,
    ) -> TemporalOutputPlan:
        native = max(2, int(native_frames))
        fps = max(1, int(requested_fps or 24))

        duration = None
        if requested_duration_seconds is not None:
            try:
                parsed = float(requested_duration_seconds)
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0:
                duration = parsed

        if duration is None:
            delivery_frames = native
            target_duration = delivery_frames / fps
            strategy = "DIRECT"
            source_fps = None
        else:
            delivery_frames = max(2, round(duration * fps))
            target_duration = delivery_frames / fps
            if delivery_frames == native:
                strategy = "DIRECT"
                source_fps = None
            elif delivery_frames > native and native >= 3:
                strategy = "MOTION_INTERPOLATION"
                # minterpolate retarde sa sortie d'environ deux frames source.
                # Le calcul + -frames:v donne une cible exacte, avec validation
                # ffprobe et fallback si le build ffmpeg se comporte autrement.
                source_fps = max(0.001, (native - 2) / target_duration)
            else:
                strategy = "LINEAR_RESAMPLE"
                source_fps = None

        return TemporalOutputPlan(
            native_frames=native,
            requested_duration_seconds=duration,
            delivery_fps=fps,
            delivery_frames=delivery_frames,
            target_duration_seconds=target_duration,
            source_fps_for_motion_interpolation=source_fps,
            strategy=strategy,
            tolerance_seconds=max(0.08, 1.5 / fps),
        )
