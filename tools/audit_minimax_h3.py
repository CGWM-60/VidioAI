#!/usr/bin/env python3
"""Audit réseau léger d'un repository ModularPipeline Hugging Face.

Par défaut, cible le repository officiel MiniMax H3. Aucun poids n'est téléchargé:
l'outil lit l'API metadata et les JSON légers uniquement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any


DEFAULT_REPO = "MiniMaxAI/MiniMax-H3"


def get_json(url: str, token: str | None = None) -> dict[str, Any]:
    headers = {
        "User-Agent": "VidioAI-H3-Audit/2026.08.11-12",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Réponse JSON inattendue pour {url}")
    return payload


def get_text(url: str, token: str | None = None) -> str | None:
    headers = {
        "User-Agent": "VidioAI-H3-Audit/2026.08.11-12",
        "Accept": "application/json,text/plain,*/*",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def component_loading(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list) and len(raw) >= 3 and isinstance(raw[2], dict):
        return raw[2]
    if isinstance(raw, dict):
        nested = raw.get("loading_specs_dict")
        if isinstance(nested, dict):
            return nested
        return raw
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    repo = args.repo.strip("/")
    quoted = urllib.parse.quote(repo, safe="/")
    api = f"https://huggingface.co/api/models/{quoted}?expand[]=siblings"
    resolve = (
        f"https://huggingface.co/{quoted}/resolve/"
        f"{urllib.parse.quote(args.revision, safe='')}"
    )

    report: dict[str, Any] = {
        "repository": repo,
        "requested_revision": args.revision,
        "discovered": False,
        "downloadable": False,
        "model_index": False,
        "modular_model_index": False,
        "architecture": None,
        "siblings_count": 0,
        "safetensors_count": 0,
        "known_size_bytes": 0,
        "top_level": [],
        "components": {},
        "workflows": [],
        "errors": [],
    }

    try:
        metadata = get_json(api, token)
    except Exception as error:
        report["errors"].append(
            f"HF_METADATA_FAILED: {type(error).__name__}: {error}"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    report["discovered"] = True
    report["downloadable"] = not bool(metadata.get("private"))
    report["resolved_revision"] = metadata.get("sha")
    report["gated"] = metadata.get("gated")
    report["private"] = metadata.get("private")
    report["pipeline_tag"] = metadata.get("pipeline_tag")
    report["library_name"] = metadata.get("library_name")

    siblings = metadata.get("siblings") or []
    paths: list[str] = []
    total = 0
    safe_count = 0
    for item in siblings:
        if not isinstance(item, dict):
            continue
        path = item.get("rfilename") or item.get("path")
        if not isinstance(path, str):
            continue
        paths.append(path)
        if path.endswith(".safetensors"):
            safe_count += 1
        size = item.get("size")
        if not isinstance(size, int):
            lfs = item.get("lfs")
            if isinstance(lfs, dict):
                size = lfs.get("size")
        if isinstance(size, int) and size > 0:
            total += size

    report["siblings_count"] = len(paths)
    report["safetensors_count"] = safe_count
    report["known_size_bytes"] = total
    report["model_index"] = "model_index.json" in paths
    report["modular_model_index"] = "modular_model_index.json" in paths
    report["top_level"] = sorted(
        {
            path.split("/", 1)[0]
            for path in paths
            if path
        }
    )

    for manifest_name in (
        "modular_model_index.json",
        "model_index.json",
    ):
        raw = get_text(f"{resolve}/{manifest_name}", token)
        if raw is None:
            continue
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as error:
            report["errors"].append(
                f"{manifest_name}: JSON_INVALID: {error}"
            )
            continue
        if not isinstance(manifest, dict):
            continue

        report[f"{manifest_name}_content"] = manifest
        if manifest_name == "modular_model_index.json":
            report["architecture"] = (
                manifest.get("_class_name")
                or manifest.get("_blocks_class_name")
            )
            components = {}
            for name, value in manifest.items():
                if str(name).startswith("_"):
                    continue
                loading = component_loading(value)
                components[name] = {
                    "repository": loading.get(
                        "pretrained_model_name_or_path"
                    ),
                    "subfolder": loading.get("subfolder"),
                    "revision": loading.get("revision"),
                    "variant": loading.get("variant"),
                    "type_hint": loading.get("type_hint"),
                }
            report["components"] = components

            raw_workflows = (
                manifest.get("_workflows")
                or manifest.get("workflows")
                or []
            )
            if isinstance(raw_workflows, dict):
                report["workflows"] = sorted(raw_workflows)
            elif isinstance(raw_workflows, list):
                report["workflows"] = [
                    str(value) for value in raw_workflows
                ]

    expected = {
        "transformer",
        "transformer_ref",
        "text_encoder",
        "vae",
        "audio_vae",
        "scheduler",
        "audio_scheduler",
    }
    present = set(report["top_level"])
    report["expected_h3_subfolders"] = {
        name: name in present for name in sorted(expected)
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Repository       : {report['repository']}")
    print(f"DISCOVERED       : {'✅' if report['discovered'] else '❌'}")
    print(f"DOWNLOADABLE     : {'✅' if report['downloadable'] else '❌'}")
    print(f"Revision         : {report.get('resolved_revision') or 'inconnue'}")
    print(
        "Manifest         : "
        f"model_index={'oui' if report['model_index'] else 'non'} · "
        f"modular_model_index={'oui' if report['modular_model_index'] else 'non'}"
    )
    print(f"Architecture     : {report.get('architecture') or 'inconnue'}")
    print(
        f"Safetensors      : {report['safetensors_count']} · "
        f"taille connue {report['known_size_bytes'] / 1024**3:.2f} GiB"
    )
    print("Sous-dossiers    : " + ", ".join(report["top_level"]))
    if report["components"]:
        print("Composants:")
        for name, component in sorted(report["components"].items()):
            print(
                f"  - {name}: repo={component.get('repository') or 'local'} "
                f"subfolder={component.get('subfolder') or '-'}"
            )
    if report["workflows"]:
        print("Workflows        : " + ", ".join(report["workflows"]))
    if report["errors"]:
        print("Erreurs:")
        for error in report["errors"]:
            print(f"  - {error}")

    print()
    print("JSON complet: relancer avec --json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
