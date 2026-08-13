#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

def patch_file(rel, old, new):
    path = root / rel
    if not path.is_file():
        raise SystemExit(f"Fichier introuvable: {rel}")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{rel}: motif attendu 1 fois, trouvé {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"OK  {rel}")

# release-install.sh : dériver le nom d'étape du numéro.
patch_file(
    "deploy/release-install.sh",
    """stage() {
  CURRENT_STEP=${1:?numéro d'étape requis}
  CURRENT_STAGE=${2:?nom d'étape requis}
  printf '[%s/7] %s\\n' "${CURRENT_STEP}" "${CURRENT_STAGE}"
}
""",
    """stage() {
  CURRENT_STEP=${1:?numéro d'étape requis}
  case "${CURRENT_STEP}" in
    1) CURRENT_STAGE=Version ;;
    2) CURRENT_STAGE=Download ;;
    3) CURRENT_STAGE=Verify ;;
    4) CURRENT_STAGE=Build ;;
    5) CURRENT_STAGE=Install ;;
    6) CURRENT_STAGE=Start ;;
    7) CURRENT_STAGE=Healthcheck ;;
    *) fail "Étape release inconnue: ${CURRENT_STEP}" ;;
  esac
  printf '[%s/7] %s\\n' "${CURRENT_STEP}" "${CURRENT_STAGE}"
}
""",
)

path = root / "deploy/release-install.sh"
text = path.read_text(encoding="utf-8")
repls = {
    "stage 1 Version": "stage 1",
    "stage 2 Download": "stage 2",
    "stage 3 Verify": "stage 3",
    "stage 4 Build": "stage 4",
    "stage 5 Install": "stage 5",
    "stage 6 Start": "stage 6",
    "stage 7 Healthcheck": "stage 7",
}
for old, new in repls.items():
    if text.count(old) != 1:
        raise SystemExit(f"deploy/release-install.sh: motif {old!r} attendu 1 fois")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("OK  deploy/release-install.sh stages")

# Enlever le tracing temporaire de CI maintenant que la cause est identifiée.
patch_file(
    ".github/workflows/ci.yml",
    """          PS4='+ ${BASH_SOURCE}:${LINENO}: ' bash -x deploy/tests/test-release-install.sh
""",
    """          bash deploy/tests/test-release-install.sh
""",
)

print()
print("Correctif V3 appliqué.")
print("Validation locale conseillée:")
print("  bash -n deploy/release-install.sh deploy/tests/test-release-install.sh")
print("  bash deploy/tests/test-release-install.sh")
