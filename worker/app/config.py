"""Configuration du worker, exclusivement pilotée par l'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Valeurs immuables afin qu'un job ne change pas de profil en cours de route."""

    app_env: str
    gpu_required: bool
    models_dir: Path
    work_dir: Path
    outputs_dir: Path
    hf_home: Path
    worker_token: str | None
    minimum_weights_bytes: int
    default_model_id: str
    default_repository: str

    @classmethod
    def from_env(cls) -> "Settings":
        app_env = os.getenv("APP_ENV", "LOCAL").strip().upper()
        # Les valeurs locales restent dans le projet courant. En conteneur,
        # Docker fournit explicitement /models, /work et /outputs.
        data_dir = Path(os.getenv("VIDIOAI_DATA_DIR", str(Path.cwd() / "data")))
        return cls(
            app_env=app_env,
            gpu_required=_bool_env("GPU_REQUIRED", app_env == "GPU_PRODUCTION"),
            models_dir=Path(
                os.getenv("VIDIOAI_MODELS_DIR", str(data_dir / "models"))
            ),
            work_dir=Path(os.getenv("VIDIOAI_WORK_DIR", str(data_dir / "work"))),
            outputs_dir=Path(
                os.getenv("VIDIOAI_OUTPUTS_DIR", str(data_dir / "outputs"))
            ),
            hf_home=Path(
                os.getenv("HF_HOME", str(data_dir / "models" / "huggingface"))
            ),
            worker_token=os.getenv("VIDIOAI_WORKER_TOKEN") or None,
            minimum_weights_bytes=int(
                os.getenv("VIDIOAI_MINIMUM_WEIGHTS_BYTES", str(10 * 1024 * 1024))
            ),
            default_model_id=os.getenv(
                "VIDIOAI_DEFAULT_AI_MODEL_ID", "stable-image-core"
            ),
            default_repository=os.getenv(
                "VIDIOAI_DEFAULT_AI_REPOSITORY", "stabilityai/sd-turbo"
            ),
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.models_dir,
            self.work_dir,
            self.outputs_dir,
            self.hf_home,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def configuration_errors(self) -> list[str]:
        errors: list[str] = []
        if self.app_env not in {"LOCAL", "GPU_PRODUCTION"}:
            errors.append("APP_ENV doit être LOCAL ou GPU_PRODUCTION")
        if self.gpu_required and self.app_env != "GPU_PRODUCTION":
            errors.append("GPU_REQUIRED=true exige APP_ENV=GPU_PRODUCTION")
        if self.app_env == "GPU_PRODUCTION" and (
            not self.worker_token
            or len(self.worker_token) < 32
            or self.worker_token.startswith("replace-with")
        ):
            errors.append(
                "VIDIOAI_WORKER_TOKEN doit être un secret aléatoire d'au moins 32 caractères"
            )
        return errors
