#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path.cwd()

REQUIRED = [
    ROOT / 'compose.production.yml',
    ROOT / '.env.production.example',
    ROOT / '.github/workflows/ci.yml',
    ROOT / 'deploy/tests/test-production-compose-contract.sh',
    ROOT / 'deploy/release-install.sh',
    ROOT / 'worker/app/engines/comfyui.py',
    ROOT / 'worker/app/generation/preflight.py',
    ROOT / 'worker/app/runtime.py',
    ROOT / 'backend/src/model_pack_registry.rs',
]


def die(msg: str) -> None:
    print(f'ERROR: {msg}', file=sys.stderr)
    raise SystemExit(1)


def read(path: Path) -> str:
    if not path.is_file():
        die(f'fichier absent: {path.relative_to(ROOT)}')
    return path.read_text(encoding='utf-8')


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')
    print(f'OK  {path.relative_to(ROOT)}')


def replace_once(content: str, old: str, new: str, label: str) -> str:
    count = content.count(old)
    if count == 0:
        # idempotence: if replacement already present, accept it.
        if new in content:
            return content
        die(f'pattern introuvable pour {label}')
    if count != 1:
        die(f'pattern ambigu ({count}) pour {label}')
    return content.replace(old, new, 1)


for path in REQUIRED:
    if not path.exists():
        die(f'lance ce script depuis la racine de VidioAI; absent: {path.name}')

# ---------------------------------------------------------------------------
# 1) Production: ComfyUI enabled by default + real VRAM reserve.
# ---------------------------------------------------------------------------
p = ROOT / '.env.production.example'
s = read(p)
s = replace_once(s, 'COMPOSE_PROFILES=\n', 'COMPOSE_PROFILES=comfyui\n', 'COMPOSE_PROFILES default')
if 'VIDIOAI_COMFYUI_RESERVE_VRAM_GB=' not in s:
    s = s.replace('COMFYUI_URL=http://comfyui:8188\n', 'COMFYUI_URL=http://comfyui:8188\nVIDIOAI_COMFYUI_RESERVE_VRAM_GB=12\n', 1)
write(p, s)

p = ROOT / 'compose.production.yml'
s = read(p)
s = replace_once(
    s,
    '    command: ["--output-directory", "/work"]\n',
    '    # ComfyUI garde sa gestion mémoire dynamique native, mais une réserve\n'
    '    # VRAM réelle est imposée au moteur afin d\'éviter qu\'un chargement\n'
    '    # compatible remplisse 100 % de la carte avant l\'inférence.\n'
    '    command: ["--output-directory", "/work", "--reserve-vram", "${VIDIOAI_COMFYUI_RESERVE_VRAM_GB:-12}"]\n',
    'ComfyUI reserve-vram',
)
write(p, s)

# ---------------------------------------------------------------------------
# 2) ComfyUI: expose actual installed node types via /object_info.
# ---------------------------------------------------------------------------
p = ROOT / 'worker/app/engines/comfyui.py'
s = read(p)
anchor = '''    def queue(self) -> dict[str, Any]:\n        value = self._request("GET", "/queue")\n        return value if isinstance(value, dict) else {}\n'''
insert = '''    def object_info(self) -> dict[str, Any]:\n        \"\"\"Return the node definitions actually installed in this ComfyUI instance.\"\"\"\n        value = self._request("GET", "/object_info")\n        if not isinstance(value, dict):\n            raise EngineError(\n                "ComfyUI /object_info a retourné une réponse invalide.",\n                code="COMFYUI_OBJECT_INFO_INVALID",\n                retryable=True,\n            )\n        return value\n\n    def node_types(self) -> set[str]:\n        return {str(name) for name in self.object_info().keys() if str(name).strip()}\n\n'''
if 'def object_info(self)' not in s:
    s = replace_once(s, anchor, insert + anchor, 'ComfyUI object_info')
write(p, s)

# ---------------------------------------------------------------------------
# 3) Runtime health: fail closed if ComfyUI node inventory is unavailable.
# ---------------------------------------------------------------------------
p = ROOT / 'worker/app/runtime.py'
s = read(p)
old = '''            return self._comfyui.health()\n        return {\n            "ready": bool(loaded is not None and loaded.pipeline is not None),\n'''
new = '''            health = self._comfyui.health()\n            if not health.get("ready"):\n                return health\n            try:\n                health["available_node_types"] = sorted(self._comfyui.node_types())\n            except EngineError as error:\n                return {\n                    **health,\n                    "ready": False,\n                    "error": str(error),\n                    "error_code": error.code,\n                }\n            return health\n        return {\n            "ready": bool(loaded is not None and loaded.pipeline is not None),\n'''
s = replace_once(s, old, new, 'runtime ComfyUI node inventory')
write(p, s)

# ---------------------------------------------------------------------------
# 4) Preflight: validate class_type against live /object_info before READY.
# ---------------------------------------------------------------------------
p = ROOT / 'worker/app/generation/preflight.py'
s = read(p)
old = '''                checks.append(PreflightCheck(name="workflow", ok=True, message="Workflow validé."))\n            except WorkflowValidationError as error:\n'''
new = '''                checks.append(PreflightCheck(name="workflow", ok=True, message="Workflow validé."))\n                available_node_types = health.get("available_node_types")\n                if pack.engine == "comfyui":\n                    if not isinstance(available_node_types, list):\n                        check(\n                            "comfyui_nodes",\n                            False,\n                            "NODE_MISSING",\n                            "Inventaire /object_info ComfyUI indisponible.",\n                            retryable=True,\n                        )\n                    else:\n                        required_node_types = {\n                            str(node.get("class_type"))\n                            for node in built.workflow.values()\n                            if isinstance(node, dict) and node.get("class_type")\n                        }\n                        missing_node_types = sorted(\n                            required_node_types - {str(value) for value in available_node_types}\n                        )\n                        check(\n                            "comfyui_nodes",\n                            not missing_node_types,\n                            "NODE_MISSING",\n                            (\n                                "Tous les nodes ComfyUI requis sont installés."\n                                if not missing_node_types\n                                else "Nodes ComfyUI absents: " + ", ".join(missing_node_types)\n                            ),\n                        )\n            except WorkflowValidationError as error:\n'''
s = replace_once(s, old, new, 'preflight live node validation')
write(p, s)

# ---------------------------------------------------------------------------
# 5) ModelPack bundled versioning: content-derived patch version.
#    This cleanly detects a changed pack even when schema_version stays at 1.
# ---------------------------------------------------------------------------
p = ROOT / 'backend/src/model_pack_registry.rs'
s = read(p)
old = '''        for pack in bundled {\n            let version = format!("{}.0.0", pack.schema_version);\n            if index\n'''
new = '''        for pack in bundled {\n            // schema_version décrit le format du contrat, pas la version du pack.\n            // Le patch semver est dérivé du contenu canonique : modifier un pack\n            // crée donc automatiquement une nouvelle version visible/rollbackable\n            // sans casser la compatibilité du schema.\n            let fingerprint = pack_sha256(&pack)?;\n            let patch = u64::from_str_radix(&fingerprint[..8], 16)\n                .map_err(|error| format!("MODEL_PACK_VERSION_INVALID: {error}"))?;\n            let version = format!("{}.0.{patch}", pack.schema_version);\n            if index\n'''
s = replace_once(s, old, new, 'content-derived ModelPack version')
write(p, s)

# ---------------------------------------------------------------------------
# 6) Release installer: activate ComfyUI by default for upgraded installs too.
# ---------------------------------------------------------------------------
p = ROOT / 'deploy/release-install.sh'
s = read(p)
old = '''VERSION=${REQUESTED_VERSION}\nexport VIDIOAI_VERSION="${VERSION}"\nCOMFYUI_IMAGE=${VIDIOAI_COMFYUI_IMAGE:-ghcr.io/lecode-official/comfyui-docker@sha256:e27739fc19d577d694ea99846a6c602e06dac963bebb2f056e22d97d19c392dd}\n'''
new = '''VERSION=${REQUESTED_VERSION}\nexport VIDIOAI_VERSION="${VERSION}"\n# Les releases actuelles utilisent ComfyUI comme moteur interne principal pour\n# les packs ComfyUI. Une ancienne .env sans profil ne doit pas le laisser éteint.\nif [[ -z "${COMPOSE_PROFILES:-}" ]]; then\n  COMPOSE_PROFILES=comfyui\n  export COMPOSE_PROFILES\nfi\nVIDIOAI_COMFYUI_RESERVE_VRAM_GB=${VIDIOAI_COMFYUI_RESERVE_VRAM_GB:-12}\nexport VIDIOAI_COMFYUI_RESERVE_VRAM_GB\nCOMFYUI_IMAGE=${VIDIOAI_COMFYUI_IMAGE:-ghcr.io/lecode-official/comfyui-docker@sha256:e27739fc19d577d694ea99846a6c602e06dac963bebb2f056e22d97d19c392dd}\n'''
s = replace_once(s, old, new, 'release ComfyUI defaults')
old2 = '''upsert_env VIDIOAI_VERSION "${VERSION}"\nupsert_env VIDIOAI_REGISTRY "${REGISTRY}"\n\nif [[ "${VIDIOAI_RUN_BOOTSTRAP:-false}" == "true" ]]; then\n'''
new2 = '''upsert_env VIDIOAI_VERSION "${VERSION}"\nupsert_env VIDIOAI_REGISTRY "${REGISTRY}"\nupsert_env COMPOSE_PROFILES "${COMPOSE_PROFILES}"\nupsert_env VIDIOAI_COMFYUI_RESERVE_VRAM_GB "${VIDIOAI_COMFYUI_RESERVE_VRAM_GB}"\n\nif [[ "${VIDIOAI_RUN_BOOTSTRAP:-false}" == "true" ]]; then\n'''
s = replace_once(s, old2, new2, 'persist ComfyUI release defaults')
write(p, s)

# ---------------------------------------------------------------------------
# 7) CI: execute release installer test and make compose contract robust to
#    Docker Compose omitting read_only=false from JSON.
# ---------------------------------------------------------------------------
p = ROOT / '.github/workflows/ci.yml'
s = read(p)
old = '''          bash deploy/tests/test-build-release-immutability.sh\n          bash deploy/tests/test-production-compose-contract.sh\n'''
new = '''          bash deploy/tests/test-build-release-immutability.sh\n          bash deploy/tests/test-production-compose-contract.sh\n          bash deploy/tests/test-release-install.sh\n'''
s = replace_once(s, old, new, 'CI release-install test')
write(p, s)

p = ROOT / 'deploy/tests/test-production-compose-contract.sh'
s = read(p)
s = s.replace(
    '(.services.backend.volumes | any(.source == "/var/lib/vidioai/state/model-pack-registry" and .target == "/registry" and .read_only == false))',
    '(.services.backend.volumes | any(.source == "/var/lib/vidioai/state/model-pack-registry" and .target == "/registry" and ((.read_only // false) == false)))',
)
s = s.replace(
    '((.services.backend.volumes | map(select(.target == "/registry"))[0].read_only) == false)',
    '(((.services.backend.volumes | map(select(.target == "/registry"))[0].read_only) // false) == false)',
)
# Better CI diagnostics: each block reports which compose contract failed.
if 'PRODUCTION_COMPOSE_CONTRACT_DEBUG' not in s:
    s = s.replace(
        "' <<<\"${configured_with_comfy}\" >/dev/null\n\nlocal_configured=",
        "' <<<\"${configured_with_comfy}\" >/dev/null || { echo 'PRODUCTION_COMPOSE_CONTRACT_DEBUG: production+comfy failed' >&2; jq '.services | {backend,worker,comfyui}' <<<\"${configured_with_comfy}\" >&2; exit 1; }\n\nlocal_configured=",
        1,
    )
    s = s.replace(
        "' <<<\"${local_configured}\" >/dev/null\n\nif VIDIOAI_PROJECT_DIR=",
        "' <<<\"${local_configured}\" >/dev/null || { echo 'PRODUCTION_COMPOSE_CONTRACT_DEBUG: local compose failed' >&2; jq '.services | {backend,worker}' <<<\"${local_configured}\" >&2; exit 1; }\n\nif VIDIOAI_PROJECT_DIR=",
        1,
    )
write(p, s)

# ---------------------------------------------------------------------------
# 8) Add a non-GPU regression test for the new ComfyUI node inventory behavior.
# ---------------------------------------------------------------------------
test_file = ROOT / 'worker/tests/test_final_comfyui_preflight_2026_08_13.py'
if not test_file.exists():
    test_file.write_text('''from __future__ import annotations\n\nfrom app.engines.comfyui import ComfyUIEngine\n\n\nclass Response:\n    def __init__(self, payload: bytes):\n        self.payload = payload\n    def __enter__(self):\n        return self\n    def __exit__(self, *_):\n        return None\n    def read(self):\n        return self.payload\n\n\ndef test_comfyui_node_types_comes_from_object_info():\n    def opener(request, *, timeout):\n        del timeout\n        if request.full_url.endswith('/object_info'):\n            return Response(b'{"KSampler": {}, "VAEDecode": {}}')\n        raise AssertionError(request.full_url)\n\n    engine = ComfyUIEngine('http://comfy.invalid', opener=opener)\n    assert engine.node_types() == {'KSampler', 'VAEDecode'}\n''', encoding='utf-8')
    print(f'OK  {test_file.relative_to(ROOT)}')
else:
    print(f'OK  {test_file.relative_to(ROOT)} (already exists)')

print('\nFINAL FIXES APPLIED')
print('Run:')
print('  cargo fmt --check --manifest-path backend/Cargo.toml')
print('  cargo clippy --manifest-path backend/Cargo.toml --all-targets -- -D warnings')
print('  cargo test --manifest-path backend/Cargo.toml')
print('  (cd worker && pytest -q)')
print('  (cd frontend && npm test && npm run lint && npm run build)')
print('  bash deploy/tests/test-production-compose-contract.sh')
print('  bash deploy/tests/test-release-install.sh')
