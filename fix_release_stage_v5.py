#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import datetime
import os
import shutil
import subprocess
import sys
import tempfile

KNOWN_GOOD = "daad4c2ed8a4e6441335aad811031b4b26008f0a"
RELEASE_PATH = "deploy/release-install.sh"
CI_PATH = ".github/workflows/ci.yml"

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

def run(*args, check=True, capture=False):
    return subprocess.run(
        list(args),
        cwd=root,
        text=True,
        capture_output=capture,
        check=check,
    )

def fail(message):
    raise SystemExit(f"ERROR: {message}")

if not (root / ".git").is_dir():
    fail("exécute ce script depuis la racine du repository VidioAI")

release = root / RELEASE_PATH
if not release.exists():
    fail(f"{RELEASE_PATH} introuvable")

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
backup_dir = Path(tempfile.gettempdir()) / "vidioai-release-repair"
backup_dir.mkdir(parents=True, exist_ok=True)
backup = backup_dir / f"release-install.sh.{stamp}.bak"
shutil.copy2(release, backup)
print(f"BACKUP  {backup}")

probe = run(
    "git", "cat-file", "-e", f"{KNOWN_GOOD}:{RELEASE_PATH}",
    check=False,
    capture=True,
)
if probe.returncode != 0:
    print(f"FETCH   commit {KNOWN_GOOD}")
    fetch = run(
        "git", "fetch", "origin", KNOWN_GOOD,
        check=False,
        capture=True,
    )
    if fetch.returncode != 0:
        sys.stderr.write(fetch.stderr)
        fail(f"impossible de récupérer le commit connu {KNOWN_GOOD}")

show = run(
    "git", "show", f"{KNOWN_GOOD}:{RELEASE_PATH}",
    check=False,
    capture=True,
)
if show.returncode != 0:
    sys.stderr.write(show.stderr)
    fail("impossible de lire release-install.sh depuis le commit connu")

text = show.stdout

old_step = "CURRENT_STEP=${1:?numéro d'étape requis}"
new_step = "CURRENT_STEP=${1:?numero_etape_requis}"
old_stage = "CURRENT_STAGE=${2:?nom d'étape requis}"
new_stage = "CURRENT_STAGE=${2:?nom_etape_requis}"

if text.count(old_step) != 1:
    fail(f"motif CURRENT_STEP attendu 1 fois, trouvé {text.count(old_step)}")
if text.count(old_stage) != 1:
    fail(f"motif CURRENT_STAGE attendu 1 fois, trouvé {text.count(old_stage)}")

text = text.replace(old_step, new_step, 1)
text = text.replace(old_stage, new_stage, 1)

for expected in (
    "stage 1 Version",
    "stage 2 Download",
    "stage 3 Verify",
    "stage 4 Build",
    "stage 5 Install",
    "stage 6 Start",
    "stage 7 Healthcheck",
):
    if text.count(expected) != 1:
        fail(f"appel d'étape absent ou ambigu: {expected}")

tmp = release.with_name("release-install.sh.repaired.tmp")
tmp.write_text(text, encoding="utf-8")
os.chmod(tmp, 0o755)

syntax = subprocess.run(
    ["bash", "-n", str(tmp)],
    cwd=root,
    text=True,
    capture_output=True,
)
if syntax.returncode != 0:
    sys.stderr.write(syntax.stderr)
    tmp.unlink(missing_ok=True)
    fail("le fichier candidat ne passe pas bash -n; fichier actuel non remplacé")

tmp.replace(release)
os.chmod(release, 0o755)
print("OK      bash -n deploy/release-install.sh")

ci = root / CI_PATH
if ci.exists():
    ci_text = ci.read_text(encoding="utf-8")
    debug = "PS4='+ ${BASH_SOURCE}:${LINENO}: ' bash -x deploy/tests/test-release-install.sh"
    normal = "bash deploy/tests/test-release-install.sh"
    if debug in ci_text:
        ci.write_text(ci_text.replace(debug, normal, 1), encoding="utf-8")
        print("OK      bash -x temporaire retiré de ci.yml")

test_syntax = subprocess.run(
    ["bash", "-n", "deploy/tests/test-release-install.sh"],
    cwd=root,
    text=True,
    capture_output=True,
)
if test_syntax.returncode != 0:
    sys.stderr.write(test_syntax.stderr)
    fail("test-release-install.sh ne passe pas bash -n")
print("OK      bash -n deploy/tests/test-release-install.sh")

print("TEST    deploy/tests/test-release-install.sh")
test = subprocess.run(
    ["bash", "deploy/tests/test-release-install.sh"],
    cwd=root,
    text=True,
    capture_output=True,
)
sys.stdout.write(test.stdout)
sys.stderr.write(test.stderr)

if test.returncode != 0:
    fail(f"test-release-install.sh échoue encore (exit {test.returncode})")

probe_script = (
    "set -Eeuo pipefail\n"
    "CURRENT_STEP=1\n"
    "CURRENT_STAGE=Version\n"
    "stage() {\n"
    "  CURRENT_STEP=${1:?numero_etape_requis}\n"
    "  CURRENT_STAGE=${2:?nom_etape_requis}\n"
    "  printf '[%s/7] %s\\n' \"${CURRENT_STEP}\" \"${CURRENT_STAGE}\"\n"
    "}\n"
    "stage 1 Version\n"
    "stage 2 Download\n"
    "stage 3 Verify\n"
    "stage 4 Build\n"
    "stage 5 Install\n"
    "stage 6 Start\n"
    "stage 7 Healthcheck\n"
)
expected_output = (
    "[1/7] Version\n"
    "[2/7] Download\n"
    "[3/7] Verify\n"
    "[4/7] Build\n"
    "[5/7] Install\n"
    "[6/7] Start\n"
    "[7/7] Healthcheck\n"
)

probe_run = subprocess.run(
    ["bash", "-c", probe_script],
    cwd=root,
    text=True,
    capture_output=True,
)
if probe_run.returncode != 0 or probe_run.stdout != expected_output:
    sys.stderr.write(probe_run.stderr)
    sys.stderr.write(probe_run.stdout)
    fail("la vérification isolée des noms d'étapes a échoué")

print("OK      noms des 7 étapes")
print()
print("SUCCESS: correction V5 appliquée et testée.")
print()
print("Nettoyage des anciens backups créés dans le repo par V3/V4:")
print("  rm -f deploy/release-install.sh.broken-*")
print()
print("Puis:")
print("  git diff -- deploy/release-install.sh .github/workflows/ci.yml")
print("  git add deploy/release-install.sh .github/workflows/ci.yml")
print('  git commit -m "fix: correct release stage parameter parsing"')
print("  git push")
