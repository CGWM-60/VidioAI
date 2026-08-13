#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

EXPECTED_BASE = "6bf43fd75c3e709e85f4e73ce413ab4b5ccce562"
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")

def read(rel: str) -> str:
    path = root / rel
    if not path.is_file():
        die(f"fichier introuvable: {rel}")
    return path.read_text(encoding="utf-8")

def write(rel: str, value: str) -> None:
    (root / rel).write_text(value, encoding="utf-8")
    print(f"OK  {rel}")

def replace_once(text: str, old: str, new: str, rel: str) -> str:
    count = text.count(old)
    if count != 1:
        die(f"{rel}: motif attendu exactement 1 fois, trouvé {count}")
    return text.replace(old, new, 1)

def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""

head = git_head()
if head and head != EXPECTED_BASE:
    print(f"NOTE: HEAD={head}, patch préparé depuis {EXPECTED_BASE}.")
    print("      Les garde-fous de contenu empêcheront toute modification ambiguë.")

# 1. CI deploy: le test release-install coupait deploy:VERSION en deploy.
rel = "deploy/tests/test-release-install.sh"
text = read(rel)
text = replace_once(
    text,
    """[[ "$(grep -nE 'shutdown|deploy:' "${LOG_FILE}" | cut -d: -f2 | tr '\\n' ' ')" == "shutdown deploy:${VERSION} " ]]""",
    """[[ "$(grep -nE 'shutdown|deploy:' "${LOG_FILE}" | cut -d: -f2- | tr '\\n' ' ')" == "shutdown deploy:${VERSION} " ]]""",
    rel,
)
write(rel, text)

# 2. Worker: le preflight exige désormais l'inventaire /object_info ComfyUI.
rel = "worker/tests/test_worker_foundation_2026_08_13.py"
text = read(rel)
old = """    result = PreflightService(WorkflowBuilder(WORKFLOWS_DIRECTORY)).run(
        model_id="flux-test",
        pack=pack,
        capability="TEXT_TO_IMAGE",
        request=request,
        snapshot=snapshot,
        execution_plan=plan,
        engine_health=lambda: {"ready": True, "engine": "comfyui-mock"},
        dependency_errors=[],
        diagnostics={"source": "unit-test"},
    )
"""
new = """    builder = WorkflowBuilder(WORKFLOWS_DIRECTORY)
    template = builder.load(pack.workflow_for("TEXT_TO_IMAGE") or "")
    available_node_types = sorted(
        {
            str(node["class_type"])
            for node in template["workflow"].values()
            if isinstance(node, dict) and node.get("class_type")
        }
    )

    result = PreflightService(builder).run(
        model_id="flux-test",
        pack=pack,
        capability="TEXT_TO_IMAGE",
        request=request,
        snapshot=snapshot,
        execution_plan=plan,
        engine_health=lambda: {
            "ready": True,
            "engine": "comfyui-mock",
            "available_node_types": available_node_types,
        },
        dependency_errors=[],
        diagnostics={"source": "unit-test"},
    )
"""
text = replace_once(text, old, new, rel)
write(rel, text)

# 3. Backend: les tests lisent la version active au lieu de figer 1.0.0.
rel = "backend/src/model_pack_registry.rs"
text = read(rel)

old = """        let registry = VersionedPackRegistry::open(root.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        let path = artifact_path(&root, "fixture-pack", "1.0.0");
"""
new = """        let registry = VersionedPackRegistry::open(root.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        let version = registry
            .active_version("fixture-pack")
            .await
            .expect("fixture-pack actif")
            .version;
        let path = artifact_path(&root, "fixture-pack", &version);
"""
text = replace_once(text, old, new, rel)

activation_old = """            .activate("fixture-pack", "1.0.0", env!("CARGO_PKG_VERSION"))
"""
activation_new = """            .activate("fixture-pack", &version, env!("CARGO_PKG_VERSION"))
"""
activation_count = text.count(activation_old)
if activation_count != 2:
    die(f"{rel}: 2 activations fixture-pack/1.0.0 attendues, trouvé {activation_count}")
text = text.replace(activation_old, activation_new)

old = """        let publisher = VersionedPackRegistry::open(first.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        publisher
"""
new = """        let publisher = VersionedPackRegistry::open(first.clone(), vec![pack()], Some(&source), 10)
            .await
            .unwrap();
        let version = publisher
            .active_version("fixture-pack")
            .await
            .expect("fixture-pack actif")
            .version;
        publisher
"""
text = replace_once(text, old, new, rel)

old = """        assert!(
            storage
                .objects
                .read()
                .await
                .contains_key("model-packs/fixture-pack/1.0.0/workflows/fixture.json")
        );
"""
new = """        let workflow_key =
            format!("model-packs/fixture-pack/{version}/workflows/fixture.json");
        assert!(storage.objects.read().await.contains_key(&workflow_key));
"""
text = replace_once(text, old, new, rel)

text = replace_once(
    text,
    """            .ensure_local_from_storage("fixture-pack", "1.0.0", &storage)
""",
    """            .ensure_local_from_storage("fixture-pack", &version, &storage)
""",
    rel,
)

write(rel, text)

# 4. H3: pas de faux READY sans validation réelle.
rel = "model-packs/minimax-h3-diffusers-v1.json"
text = read(rel)
text = replace_once(text, '  "status": "READY",', '  "status": "EXPERIMENTAL",', rel)
write(rel, text)

# 5. CI: syntaxe du wrapper release + étape identifiable.
rel = ".github/workflows/ci.yml"
text = read(rel)
old = """          bash -n \\
            deploy/scripts/*.sh \\
            deploy/scripts/lib/*.sh \\
            deploy/tests/*.sh
"""
new = """          bash -n \\
            deploy/release-install.sh \\
            deploy/scripts/*.sh \\
            deploy/scripts/lib/*.sh \\
            deploy/tests/*.sh
"""
text = replace_once(text, old, new, rel)

old = """          bash deploy/tests/test-production-compose-contract.sh
          bash deploy/tests/test-release-install.sh
"""
new = """          bash deploy/tests/test-production-compose-contract.sh
          echo "Validation release-install..."
          bash deploy/tests/test-release-install.sh
"""
text = replace_once(text, old, new, rel)
write(rel, text)

# 6. Contrat Compose: verrouiller la réserve VRAM réelle.
rel = "deploy/tests/test-production-compose-contract.sh"
text = read(rel)
needle = """  (.services.comfyui.healthcheck.test | any(contains("http://127.0.0.1:8188/system_stats"))) and
"""
addition = """  (.services.comfyui.healthcheck.test | any(contains("http://127.0.0.1:8188/system_stats"))) and
  (.services.comfyui.command | index("--reserve-vram") != null) and
  (.services.comfyui.command | index("12") != null) and
"""
text = replace_once(text, needle, addition, rel)
write(rel, text)

print()
print("PATCH VIDIOAI FINAL V2 APPLIQUÉ")
print("- release-install test corrigé")
print("- mock /object_info Worker corrigé")
print("- tests Rust ModelPack version dynamique corrigés")
print("- H3 repassé EXPERIMENTAL")
print("- release-install.sh couvert par bash -n")
print("- --reserve-vram=12 verrouillé par le contrat Compose")
print()
print("Puis exécuter les validations non-GPU habituelles avant push.")
