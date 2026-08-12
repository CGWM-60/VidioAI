"""Mux audio natif -> AAC dans le MP4 final VidioAI.

Ce module ne fabrique jamais une piste silencieuse pour prétendre qu'un modèle
a produit de l'audio natif. Si l'utilisateur demande l'audio et que le runtime
n'en fournit pas, la génération échoue avec un code structuré.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np


class AudioOutputError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "AUDIO_OUTPUT_FAILED",
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = False


class NativeAudioMuxer:
    def __init__(self) -> None:
        for binary in ("ffmpeg", "ffprobe"):
            if shutil.which(binary) is None:
                raise AudioOutputError(
                    f"{binary} est requis pour la sortie audio native.",
                    code="MEDIA_RUNTIME_MISSING",
                    status_code=503,
                )

    @staticmethod
    def _unwrap(value: Any) -> Any:
        # Les sorties de pipeline sont souvent batchées: [waveform].
        current = value
        for _ in range(3):
            if isinstance(current, (list, tuple)) and len(current) == 1:
                current = current[0]
                continue
            break
        if hasattr(current, "detach"):
            current = current.detach()
        if hasattr(current, "cpu"):
            current = current.cpu()
        if hasattr(current, "float"):
            try:
                current = current.float()
            except Exception:
                pass
        if hasattr(current, "numpy"):
            current = current.numpy()
        return current

    @classmethod
    def _waveform(cls, value: Any) -> np.ndarray:
        value = cls._unwrap(value)
        try:
            array = np.asarray(value)
        except Exception as error:
            raise AudioOutputError(
                f"Waveform audio non convertible: {type(error).__name__}: {error}",
                code="NATIVE_AUDIO_INVALID",
                status_code=422,
            ) from error

        if array.size == 0:
            raise AudioOutputError(
                "La sortie audio native est vide.",
                code="NATIVE_AUDIO_EMPTY",
                status_code=422,
            )

        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]

        if array.ndim == 1:
            array = array[np.newaxis, :]
        elif array.ndim == 2:
            # Contrat interne: channels x samples. Si le dernier axe ressemble
            # clairement aux canaux, transpose.
            if array.shape[0] > 8 and array.shape[1] <= 8:
                array = array.T
        else:
            raise AudioOutputError(
                f"Shape audio non prise en charge: {array.shape}",
                code="NATIVE_AUDIO_SHAPE_UNSUPPORTED",
                status_code=422,
            )

        if array.shape[0] < 1 or array.shape[0] > 8:
            raise AudioOutputError(
                f"Nombre de canaux audio invalide: {array.shape[0]}",
                code="NATIVE_AUDIO_CHANNELS_INVALID",
                status_code=422,
            )

        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            scale = float(max(abs(info.min), info.max))
            array = array.astype(np.float32) / scale
        else:
            array = array.astype(np.float32, copy=False)

        if not np.isfinite(array).all():
            raise AudioOutputError(
                "La waveform audio contient NaN/Inf.",
                code="NATIVE_AUDIO_INVALID",
                status_code=422,
            )
        return np.clip(array, -1.0, 1.0)

    @staticmethod
    def _normalize_rate(value: Any) -> int:
        try:
            rate = int(value)
        except (TypeError, ValueError) as error:
            raise AudioOutputError(
                "Le runtime a produit une waveform sans sample rate exploitable.",
                code="AUDIO_SAMPLE_RATE_MISSING",
                status_code=422,
            ) from error
        if rate < 8000 or rate > 384000:
            raise AudioOutputError(
                f"Sample rate audio invalide: {rate}",
                code="AUDIO_SAMPLE_RATE_INVALID",
                status_code=422,
            )
        return rate

    @staticmethod
    def _fit_duration(
        audio: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
    ) -> np.ndarray:
        target_samples = max(1, int(round(duration_seconds * sample_rate)))
        current = audio.shape[1]
        if current == target_samples:
            return audio
        if current > target_samples:
            return audio[:, :target_samples]
        padding = np.zeros(
            (audio.shape[0], target_samples - current),
            dtype=np.float32,
        )
        return np.concatenate([audio, padding], axis=1)

    @staticmethod
    def _write_pcm16_wav(
        path: Path,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        channels = int(audio.shape[0])
        interleaved = np.rint(
            np.clip(audio.T, -1.0, 1.0) * 32767.0
        ).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(interleaved.tobytes())

    @staticmethod
    def _probe(path: Path) -> dict[str, Any]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise AudioOutputError(
                f"ffprobe a refusé le MP4 muxé: {result.stderr.strip()}",
                code="OUTPUT_MEDIA_INVALID",
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise AudioOutputError(
                "Réponse ffprobe invalide.",
                code="OUTPUT_MEDIA_INVALID",
            ) from error

        streams = payload.get("streams") or []
        video = next(
            (
                item
                for item in streams
                if item.get("codec_type") == "video"
            ),
            None,
        )
        audio = next(
            (
                item
                for item in streams
                if item.get("codec_type") == "audio"
            ),
            None,
        )
        if video is None:
            raise AudioOutputError(
                "Le MP4 final ne contient aucun stream vidéo.",
                code="OUTPUT_VIDEO_STREAM_MISSING",
            )
        if audio is None:
            raise AudioOutputError(
                "Le MP4 final ne contient aucun stream audio.",
                code="OUTPUT_AUDIO_STREAM_MISSING",
            )
        if str(video.get("codec_name") or "").lower() != "h264":
            raise AudioOutputError(
                f"Codec vidéo final inattendu: {video.get('codec_name')}",
                code="OUTPUT_VIDEO_CODEC_INVALID",
            )
        if str(audio.get("codec_name") or "").lower() != "aac":
            raise AudioOutputError(
                f"Codec audio final inattendu: {audio.get('codec_name')}",
                code="OUTPUT_AUDIO_CODEC_INVALID",
            )

        try:
            channels = int(audio.get("channels") or 0)
            sample_rate = int(audio.get("sample_rate") or 0)
        except (TypeError, ValueError):
            channels = 0
            sample_rate = 0
        if channels <= 0 or sample_rate <= 0:
            raise AudioOutputError(
                "Le stream AAC final n'expose pas canaux/sample-rate valides.",
                code="OUTPUT_AUDIO_INVALID",
            )

        duration = None
        for source in (payload.get("format") or {}, video, audio):
            raw = source.get("duration")
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed > 0:
                duration = parsed
                break

        return {
            "actual_audio": True,
            "audio_codec": "aac",
            "audio_channels": channels,
            "audio_sample_rate": sample_rate,
            "mux_duration": duration,
        }

    def mux(
        self,
        *,
        video_path: Path,
        native_audio: Any,
        audio_sample_rate: Any,
        duration_seconds: float,
    ) -> dict[str, Any]:
        if native_audio is None:
            raise AudioOutputError(
                "Audio demandé mais le modèle n'a renvoyé aucune sortie audio native.",
                code="NATIVE_AUDIO_MISSING",
                status_code=422,
            )
        if duration_seconds <= 0:
            raise AudioOutputError(
                "Durée vidéo invalide pour le mux audio.",
                code="OUTPUT_DURATION_INVALID",
                status_code=422,
            )

        waveform = self._waveform(native_audio)
        sample_rate = self._normalize_rate(audio_sample_rate)
        waveform = self._fit_duration(
            waveform,
            sample_rate,
            duration_seconds,
        )

        work = video_path.parent
        wav_path = work / f".{video_path.stem}.native-audio.wav"
        muxed = work / f".{video_path.stem}.native-audio.mp4"
        self._write_pcm16_wav(
            wav_path,
            waveform,
            sample_rate,
        )

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_path),
                    "-i",
                    str(wav_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-t",
                    f"{duration_seconds:.6f}",
                    "-movflags",
                    "+faststart",
                    str(muxed),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not muxed.is_file():
                raise AudioOutputError(
                    f"FFmpeg n'a pas pu muxer l'audio natif: {result.stderr.strip()}",
                    code="AUDIO_MUX_FAILED",
                )

            probe = self._probe(muxed)
            # Le MP4 avec audio n'est promu qu'après validation ffprobe.
            muxed.replace(video_path)
            return probe
        finally:
            wav_path.unlink(missing_ok=True)
            muxed.unlink(missing_ok=True)
