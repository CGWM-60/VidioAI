"""Normalisation des assets VidioAI et des sorties heterogenes Diffusers."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .temporal_output_planner import TemporalOutputPlanner


VIDEO_CAPABILITIES = {
    "TEXT_TO_VIDEO",
    "IMAGE_TO_VIDEO",
    "MULTI_IMAGE_TO_VIDEO",
    "START_END_IMAGE_TO_VIDEO",
    "KEYFRAMES_TO_VIDEO",
    "VIDEO_TO_VIDEO",
    "VIDEO_INPAINTING",
    "VIDEO_UPSCALE",
}


class NormalizationError(RuntimeError):
    def __init__(self, message: str, *, code: str = "GENERATION_FAILED") -> None:
        super().__init__(message)
        self.code = code


class InputNormalizer:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)

    @staticmethod
    def load_image(path: str | Path, *, mode: str = "RGB") -> Any:
        from PIL import Image

        candidate = Path(path)
        if not candidate.is_file():
            raise NormalizationError(
                f"Asset introuvable: {candidate}",
                code="INVALID_INPUT_ASSET",
            )
        with Image.open(candidate) as image:
            return image.convert(mode).copy()

    def load_video_frames(self, path: str | Path) -> list[Any]:
        candidate = Path(path)
        if not candidate.is_file():
            raise NormalizationError(
                f"Video introuvable: {candidate}",
                code="INVALID_INPUT_ASSET",
            )
        frame_dir = self.work_dir / f"input-video-{uuid.uuid4()}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", str(candidate),
                    str(frame_dir / "frame-%08d.png"),
                ],
                capture_output=True,
                text=True,
                timeout=10 * 60,
                check=False,
            )
            paths = sorted(frame_dir.glob("frame-*.png"))
            if result.returncode != 0 or not paths:
                raise NormalizationError(
                    f"Decodage video impossible: {result.stderr.strip() or 'aucune frame'}",
                    code="INVALID_INPUT_ASSET",
                )
            return [self.load_image(frame) for frame in paths]
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def normalize(self, request: dict[str, Any], accepted: set[str]) -> dict[str, Any]:
        prepared = dict(request)
        input_path = prepared.get("input_path")
        capability = str(prepared.get("capability") or "").upper()

        if isinstance(input_path, str) and input_path.strip():
            if capability in {"VIDEO_TO_VIDEO", "VIDEO_INPAINTING", "VIDEO_UPSCALE"}:
                frames = self.load_video_frames(input_path)
                prepared["input_video"] = frames if "video" in accepted else input_path
                prepared["input_frames"] = frames
            else:
                prepared["input_image"] = self.load_image(input_path)

        if prepared.get("mask_path"):
            prepared["mask_image"] = self.load_image(prepared["mask_path"], mode="L")
        if prepared.get("control_path"):
            prepared["control_image"] = self.load_image(prepared["control_path"])

        resolved: list[Any] = []
        roles: list[str] = []
        raw_images = sorted(
            [item for item in prepared.get("input_images") or [] if isinstance(item, dict)],
            key=lambda item: int(item.get("order") or 0),
        )
        for item in raw_images:
            source = item.get("source") or item.get("path") or item.get("input_path")
            if source:
                resolved.append(self.load_image(source))
                roles.append(str(item.get("role") or "reference").lower())

        if (
            raw_images
            and not resolved
            and capability in {
                "IMAGE_TO_VIDEO", "MULTI_IMAGE_TO_VIDEO",
                "START_END_IMAGE_TO_VIDEO", "KEYFRAMES_TO_VIDEO",
            }
            and not prepared.get("input_image")
        ):
            raise NormalizationError(
                "Les asset IDs doivent etre resolus vers des images avant l'appel pipeline.",
                code="INVALID_INPUT_ASSET",
            )

        if resolved:
            prepared["resolved_input_images"] = resolved
            prepared["resolved_image_roles"] = roles
        return prepared


class OutputNormalizer:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)
        self._temporal_planner = TemporalOutputPlanner()

    @staticmethod
    def normalize_frames(frames: Any) -> list[Any]:
        if frames is None:
            return []
        if hasattr(frames, "detach"):
            frames = frames.detach().float().cpu().numpy()
        ndim = getattr(frames, "ndim", None)
        if isinstance(ndim, int):
            if ndim >= 5:
                frames = frames[0]
            if getattr(frames, "ndim", None) == 4:
                shape = getattr(frames, "shape", ())
                if shape and shape[0] in {1, 3, 4} and shape[-1] not in {1, 3, 4}:
                    import numpy as np
                    frames = np.moveaxis(frames, 0, -1)
            return list(frames)
        if isinstance(frames, (list, tuple)):
            if len(frames) == 1:
                first = frames[0]
                first_ndim = getattr(first, "ndim", None)
                if isinstance(first, (list, tuple)):
                    return list(first)
                if isinstance(first_ndim, int) and first_ndim >= 4:
                    return list(first)
            return list(frames)
        return [frames]

    @staticmethod
    def extract(output: Any, *, video: bool) -> tuple[list[Any], list[Any]]:
        if isinstance(output, dict):
            images = output.get("images")
            if images is None:
                images = []
            frames = output.get("frames")
        else:
            images = getattr(output, "images", [])
            if images is None:
                images = []
            frames = getattr(output, "frames", None)
            if frames is None and isinstance(output, (list, tuple)):
                frames = output if video else None
                images = [] if video else output

        normalized_frames = OutputNormalizer.normalize_frames(frames)
        if video and not normalized_frames and images:
            normalized_frames = OutputNormalizer.normalize_frames(images)
            images = []
        return list(images), normalized_frames

    @staticmethod
    def _to_image(frame: Any) -> Any:
        from PIL import Image
        import numpy as np

        if hasattr(frame, "detach"):
            frame = frame.detach().float().cpu().numpy()
        if hasattr(frame, "convert"):
            return frame.convert("RGB")
        array = np.asarray(frame)
        if array.dtype.kind == "f":
            if array.size and float(array.min()) < 0:
                array = (array + 1.0) / 2.0
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype("uint8")
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype("uint8")
        if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array).convert("RGB")

    @staticmethod
    def probe_video(path: Path) -> dict[str, Any]:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,nb_read_frames,avg_frame_rate:format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        try:
            payload = json.loads(result.stdout)
            stream = payload["streams"][0]
            duration = float(payload.get("format", {}).get("duration") or 0)
            frames = int(stream.get("nb_frames") or stream.get("nb_read_frames") or 0)
            rate = str(stream.get("avg_frame_rate") or "0/1").split("/", maxsplit=1)
            fps = float(rate[0]) / max(1.0, float(rate[1]))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise NormalizationError("ffprobe n'a pas reconnu une video valide.") from error

        probe = {
            "codec": stream.get("codec_name"),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "duration": duration,
            "frames": frames,
            "fps": fps,
        }
        if (
            result.returncode != 0
            or probe["codec"] != "h264"
            or probe["width"] <= 0
            or probe["height"] <= 0
            or probe["duration"] <= 0
            or probe["frames"] <= 1
        ):
            raise NormalizationError(f"Video MP4 invalide apres encodage: {probe}")
        return probe

    @staticmethod
    def _encode_sequence(
        *,
        frame_dir: Path,
        output_path: Path,
        input_fps: float,
        output_fps: int,
        frames_limit: int | None = None,
        filter_graph: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{max(0.001, input_fps):.10f}",
            "-i", str(frame_dir / "frame-%08d.png"),
        ]
        if filter_graph:
            command.extend(["-vf", filter_graph])
        if frames_limit is not None:
            command.extend(["-frames:v", str(frames_limit)])
        command.extend([
            "-r", str(max(1, output_fps)),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ])
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10 * 60,
            check=False,
        )

    def _write_native_frames(self, normalized: list[Any], frame_dir: Path) -> None:
        for index, frame in enumerate(normalized):
            self._to_image(frame).save(
                frame_dir / f"frame-{index:08d}.png",
                format="PNG",
            )

    def _write_linear_resample(
        self,
        normalized: list[Any],
        frame_dir: Path,
        target_count: int,
    ) -> None:
        from PIL import Image

        source = [self._to_image(frame) for frame in normalized]
        if target_count <= 1:
            raise NormalizationError("Le reechantillonnage video exige au moins deux frames.")
        if len(source) == 1:
            source = [source[0], source[0]]

        for index in range(target_count):
            position = index * (len(source) - 1) / (target_count - 1)
            low = int(math.floor(position))
            high = min(len(source) - 1, low + 1)
            alpha = position - low
            image = (
                source[low].copy()
                if high == low or alpha <= 1e-9
                else Image.blend(source[low], source[high], alpha)
            )
            image.save(frame_dir / f"frame-{index:08d}.png", format="PNG")

    @staticmethod
    def _probe_matches_plan(
        probe: dict[str, Any],
        *,
        delivery_frames: int,
        delivery_fps: int,
        target_duration: float,
        tolerance: float,
    ) -> bool:
        return (
            int(probe.get("frames") or 0) == delivery_frames
            and abs(float(probe.get("fps") or 0) - delivery_fps) <= 0.05
            and abs(float(probe.get("duration") or 0) - target_duration) <= tolerance
        )

    def write_video(
        self,
        frames: Any,
        output_path: Path,
        fps: int,
        *,
        duration_seconds: float | int | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_frames(frames)
        if len(normalized) <= 1:
            raise NormalizationError("Une generation video doit contenir plus d'une frame.")
        if output_path.suffix.lower() != ".mp4":
            raise NormalizationError(
                "Une sortie video doit utiliser l'extension .mp4.",
                code="INVALID_OUTPUT_PATH",
            )

        plan = self._temporal_planner.plan(
            native_frames=len(normalized),
            requested_duration_seconds=duration_seconds,
            requested_fps=fps,
        )

        native_dir = self.work_dir / f"output-video-native-{uuid.uuid4()}"
        native_dir.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f"{output_path.stem}.tmp.mp4")
        strategy_used = plan.strategy

        try:
            self._write_native_frames(normalized, native_dir)

            if plan.strategy == "DIRECT":
                result = self._encode_sequence(
                    frame_dir=native_dir,
                    output_path=temporary,
                    input_fps=plan.delivery_fps,
                    output_fps=plan.delivery_fps,
                    frames_limit=plan.delivery_frames,
                )
            elif plan.strategy == "MOTION_INTERPOLATION":
                result = self._encode_sequence(
                    frame_dir=native_dir,
                    output_path=temporary,
                    input_fps=plan.source_fps_for_motion_interpolation or plan.delivery_fps,
                    output_fps=plan.delivery_fps,
                    frames_limit=plan.delivery_frames,
                    filter_graph=(
                        "minterpolate="
                        f"fps={plan.delivery_fps}:"
                        "mi_mode=mci:"
                        "mc_mode=aobmc:"
                        "me_mode=bidir:"
                        "vsbmc=1"
                    ),
                )
            else:
                result = subprocess.CompletedProcess([], 1, "", "")

            probe = None
            if result.returncode == 0 and temporary.is_file():
                try:
                    probe = self.probe_video(temporary)
                except NormalizationError:
                    probe = None

            if (
                probe is None
                or not self._probe_matches_plan(
                    probe,
                    delivery_frames=plan.delivery_frames,
                    delivery_fps=plan.delivery_fps,
                    target_duration=plan.target_duration_seconds,
                    tolerance=plan.tolerance_seconds,
                )
            ):
                strategy_used = "LINEAR_RESAMPLE"
                temporary.unlink(missing_ok=True)
                resampled_dir = self.work_dir / f"output-video-resampled-{uuid.uuid4()}"
                resampled_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self._write_linear_resample(
                        normalized,
                        resampled_dir,
                        plan.delivery_frames,
                    )
                    result = self._encode_sequence(
                        frame_dir=resampled_dir,
                        output_path=temporary,
                        input_fps=plan.delivery_fps,
                        output_fps=plan.delivery_fps,
                        frames_limit=plan.delivery_frames,
                    )
                finally:
                    shutil.rmtree(resampled_dir, ignore_errors=True)

                if result.returncode != 0 or not temporary.is_file():
                    raise NormalizationError(
                        f"Encodage H.264 impossible: {result.stderr.strip()}",
                        code="OUTPUT_ENCODING_FAILED",
                    )
                probe = self.probe_video(temporary)

            if not self._probe_matches_plan(
                probe,
                delivery_frames=plan.delivery_frames,
                delivery_fps=plan.delivery_fps,
                target_duration=plan.target_duration_seconds,
                tolerance=plan.tolerance_seconds,
            ):
                raise NormalizationError(
                    "VIDEO_DURATION_MISMATCH: "
                    f"attendu {plan.target_duration_seconds:.3f}s / "
                    f"{plan.delivery_frames} frames / {plan.delivery_fps}fps, "
                    f"obtenu {probe.get('duration')}s / "
                    f"{probe.get('frames')} frames / {probe.get('fps')}fps.",
                    code="OUTPUT_DURATION_MISMATCH",
                )

            os.replace(temporary, output_path)
            return {
                **probe,
                "native_frames": len(normalized),
                "delivery_frames": plan.delivery_frames,
                "temporal_strategy": strategy_used,
                "requested_duration": plan.requested_duration_seconds,
            }
        finally:
            shutil.rmtree(native_dir, ignore_errors=True)
            temporary.unlink(missing_ok=True)

    def write_image(self, images: list[Any], output_path: Path) -> dict[str, Any]:
        if not images:
            raise NormalizationError("Le pipeline image n'a renvoye aucune image.")
        if output_path.suffix.lower() != ".png":
            raise NormalizationError(
                "Une sortie image doit utiliser l'extension .png.",
                code="INVALID_OUTPUT_PATH",
            )
        temporary = output_path.with_name(f"{output_path.stem}.tmp.png")
        image = self._to_image(images[0])
        image.save(temporary, format="PNG")
        os.replace(temporary, output_path)
        return {"width": image.width, "height": image.height}


def first_supported(accepted: set[str], *aliases: str) -> str | None:
    return next((name for name in aliases if name in accepted), None)


def assign_alias(
    kwargs: dict[str, Any],
    accepted: set[str],
    value: Any,
    *aliases: str,
) -> None:
    name = first_supported(accepted, *aliases)
    if name is not None and value is not None:
        kwargs[name] = value
