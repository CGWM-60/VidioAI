"""Installation isolée et persistante des dépendances approuvées VidioAI."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from .dependency_resolver import (
    DependencyRegistry,
    DependencyResolutionError,
    DependencySpec,
)


DEPENDENCY_SCHEMA_VERSION = 1


def runtime_fingerprint() -> str:
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "cpu")
    except Exception:
        torch_version = "missing"
        cuda_version = "missing"
    raw = (
        f"python-{sys.version_info.major}.{sys.version_info.minor}"
        f"_torch-{torch_version}_cuda-{cuda_version}"
        f"_{platform.machine()}_deps-{DEPENDENCY_SCHEMA_VERSION}"
    )
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)


class DependencyInstaller:
    def __init__(
        self,
        root: Path,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.fingerprint = runtime_fingerprint()
        self.environment = root / self.fingerprint
        self.site_packages = self.environment / "site-packages"
        self.manifest_path = self.environment / "runtime-dependencies.json"
        self._runner = runner
        self.site_packages.mkdir(parents=True, exist_ok=True)
        self.activate()

    def activate(self) -> None:
        path = str(self.site_packages)
        if path not in sys.path:
            sys.path.insert(0, path)
        importlib.invalidate_caches()

    @staticmethod
    def _version(spec: DependencySpec) -> str | None:
        try:
            return importlib.metadata.version(spec.package)
        except importlib.metadata.PackageNotFoundError:
            module = sys.modules.get(spec.import_name)
            value = getattr(module, "__version__", None) if module is not None else None
            return str(value) if value else None

    @staticmethod
    def _import(spec: DependencySpec) -> bool:
        try:
            importlib.import_module(spec.import_name)
            return True
        except (ImportError, ModuleNotFoundError, OSError):
            return False

    def _record(self, record: dict[str, Any]) -> None:
        records = self.records()
        records = [item for item in records if item.get("import_name") != record["import_name"]]
        records.append(record)
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def records(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    def status(self, import_name: str, *, required_by: str = "pipeline") -> dict[str, Any]:
        spec = DependencyRegistry.resolve(import_name)
        available = self._import(spec)
        return {
            "import_name": spec.import_name,
            "package": spec.package,
            "version": self._version(spec) if available else spec.version,
            "status": "AVAILABLE" if available else "REQUIRED",
            "source": "runtime" if available else "pypi",
            "automatic": False,
            "required_by": required_by,
            "runtime_fingerprint": self.fingerprint,
        }

    def ensure(
        self,
        import_name: str,
        *,
        required_by: str = "pipeline",
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        spec = DependencyRegistry.resolve(import_name)
        if self._import(spec):
            actual_version = self._version(spec)
            if actual_version is not None and actual_version != spec.version:
                raise DependencyResolutionError(
                    f"{spec.package} {actual_version} est chargé, version {spec.version} requise.",
                    code="DEPENDENCY_VERSION_CONFLICT",
                    dependency=spec.import_name,
                )
            record = self.status(spec.import_name, required_by=required_by)
            self._record(record)
            return record
        if platform.system() != "Linux":
            if progress is not None:
                progress("BLOCKED")
            raise DependencyResolutionError(
                f"L'installation automatique de {spec.package} est réservée au Worker Linux.",
                code="DEPENDENCY_PLATFORM_UNSUPPORTED",
                dependency=spec.import_name,
            )
        try:
            with tempfile.TemporaryDirectory(
                prefix="download-", dir=self.environment
            ) as download_directory:
                if progress is not None:
                    progress("DOWNLOADING")
                self._runner(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-deps",
                        "--only-binary=:all:",
                        "--dest",
                        download_directory,
                        f"{spec.package}=={spec.version}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15 * 60,
                )
                wheels = sorted(Path(download_directory).glob("*.whl"))
                if len(wheels) != 1:
                    raise OSError(
                        f"pip download a produit {len(wheels)} wheel(s), une attendue"
                    )
                if progress is not None:
                    progress("INSTALLING")
                self._runner(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-deps",
                        "--no-index",
                        "--target",
                        str(self.site_packages),
                        str(wheels[0]),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15 * 60,
                )
        except (subprocess.SubprocessError, OSError) as error:
            if progress is not None:
                progress("FAILED")
            detail = getattr(error, "stderr", None) or str(error)
            raise DependencyResolutionError(
                f"Installation de {spec.package} impossible: {detail}",
                code="DEPENDENCY_INSTALL_FAILED",
                dependency=spec.import_name,
            ) from error
        self.activate()
        if not self._import(spec):
            raise DependencyResolutionError(
                f"Import {spec.import_name} impossible après installation de {spec.package}.",
                code="DEPENDENCY_INSTALL_FAILED",
                dependency=spec.import_name,
            )
        actual_version = self._version(spec)
        if actual_version is not None and actual_version != spec.version:
            raise DependencyResolutionError(
                f"{spec.package} {actual_version} a été importé, version {spec.version} requise.",
                code="DEPENDENCY_VERSION_CONFLICT",
                dependency=spec.import_name,
            )
        record = {
            "import_name": spec.import_name,
            "package": spec.package,
            "version": actual_version or spec.version,
            "status": "INSTALLED",
            "source": "pypi",
            "automatic": True,
            "required_by": required_by,
            "runtime_fingerprint": self.fingerprint,
        }
        self._record(record)
        if progress is not None:
            progress("INSTALLED")
        return record
