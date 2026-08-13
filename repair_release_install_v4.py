#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import shutil
import datetime
import os

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
release_rel = "deploy/release-install.sh"
ci_rel = ".github/workflows/ci.yml"

def run(*args, check=True, capture=False):
    return subprocess.run(
        list(args),
        cwd=root,
        check=check,
        text=True,
        capture_output=capture,
    )

def git_show(path: str) -> str:
    p = run("git", "show", f"HEAD:{path}", capture=True)
    return p.stdout

def die(msg: str):
    raise SystemExit(f"ERROR: {msg}")

if not (root / ".git").exists():
    die("lance ce script depuis la racine du repository VidioAI")

release_path = root / release_rel
ci_path = root / ci_rel

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if release_path.exists():
    backup = release_path.with_name(f"release-install.sh.broken-{stamp}")
    shutil.copy2(release_path, backup)
    print(f"BACKUP  {backup.relative_to(root)}")

text = git_show(release_rel)

stage_pattern = re.compile(
    "stage\\(\\) \\{\\n.*?\\n\\}\\n\\nfail\\(\\) \\{",
    re.DOTALL,
)

replacement = '''stage() {
  CURRENT_STEP=${1:?numéro d'étape requis}
  case "${CURRENT_STEP}" in
    1) CURRENT_STAGE=Version ;;
    2) CURRENT_STAGE=Download ;;
    3) CURRENT_STAGE=Verify ;;
    4) CURRENT_STAGE=Build ;;
    5) CURRENT_STAGE=Install ;;
    6) CURRENT_STAGE=Start ;;
    7) CURRENT_STAGE=Healthcheck ;;
    *) printf 'Étape release inconnue: %s\\n' "${CURRENT_STEP}" >&2; return 1 ;;
  esac
  printf '[%s/7] %s\\n' "${CURRENT_STEP}" "${CURRENT_STAGE}"
}

fail() {'''

text, count = stage_pattern.subn(replacement, text, count=1)
if count != 1:
    die(f"fonction stage() introuvable ou ambiguë dans HEAD ({count} correspondance)")

expected_calls = {
    "stage 1 Version": "stage 1",
    "stage 2 Download": "stage 2",
    "stage 3 Verify": "stage 3",
    "stage 4 Build": "stage 4",
    "stage 5 Install": "stage 5",
    "stage 6 Start": "stage 6",
    "stage 7 Healthcheck": "stage 7",
}
for old, new in expected_calls.items():
    n = text.count(old)
    if n != 1:
        die(f"{old!r}: attendu 1 fois dans HEAD, trouvé {n}")
    text = text.replace(old, new, 1)

release_path.write_text(text, encoding="utf-8")
os.chmod(release_path, 0o755)

syntax = run("bash", "-n", release_rel, check=False, capture=True)
if syntax.returncode != 0:
    print(syntax.stderr, file=sys.stderr)
    die("le fichier réparé ne passe pas bash -n")
print("OK      bash -n deploy/release-install.sh")

if ci_path.exists():
    ci = ci_path.read_text(encoding="utf-8")
    debug = "          PS4='+ ${BASH_SOURCE}:${LINENO}: ' bash -x deploy/tests/test-release-install.sh"
    normal = "          bash deploy/tests/test-release-install.sh"
    if debug in ci:
        ci = ci.replace(debug, normal, 1)
        ci_path.write_text(ci, encoding="utf-8")
        print("OK      CI: bash -x temporaire retiré")
    else:
        print("INFO    CI: pas de bash -x temporaire à retirer")

test_syntax = run(
    "bash", "-n",
    "deploy/tests/test-release-install.sh",
    check=False,
    capture=True,
)
if test_syntax.returncode != 0:
    print(test_syntax.stderr, file=sys.stderr)
    die("test-release-install.sh ne passe pas bash -n")
print("OK      bash -n deploy/tests/test-release-install.sh")

print("TEST    deploy/tests/test-release-install.sh")
test = run(
    "bash", "deploy/tests/test-release-install.sh",
    check=False,
    capture=True,
)
sys.stdout.write(test.stdout)
sys.stderr.write(test.stderr)

if test.returncode != 0:
    die(f"test-release-install.sh échoue encore (exit {test.returncode})")

print()
print("SUCCESS: release-install.sh réparé et test d'orchestration vert.")
print("Tu peux maintenant faire:")
print("  git diff -- deploy/release-install.sh .github/workflows/ci.yml")
print("  git add deploy/release-install.sh .github/workflows/ci.yml")
print('  git commit -m "fix: repair release install stage handling"')
print("  git push")
