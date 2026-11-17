#!/usr/bin/env bash
# Set up MobileGym for this code: clone it at the pinned commit (unless $MOBILEGYM_DIR already holds it), install its
# Node packages without lifecycle scripts, build the production simulator without the optional dataset, and build the
# icon name table of the text view from its lucide-react package.
# Needs: git, Node 22 (node and npm on PATH), and the Python environment of requirements/sim.txt (for Playwright's
# Chromium). Usage: bash scripts/setup_sim.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
: "${MOBILEGYM_DIR:?source configs/paths.env first}"
URL=https://github.com/Purewhiter/mobilegym
COMMIT=5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38
ICONS_MD5=a2eee712b94ed5997407db4363948140

if [ ! -d "$MOBILEGYM_DIR/.git" ]; then
  git clone "$URL" "$MOBILEGYM_DIR"
fi
git -C "$MOBILEGYM_DIR" checkout --quiet "$COMMIT"
[ "$(git -C "$MOBILEGYM_DIR" rev-parse HEAD)" = "$COMMIT" ] || { echo "MobileGym is not at $COMMIT"; exit 1; }
[ -z "$(git -C "$MOBILEGYM_DIR" status --short)" ] || echo "warning: the MobileGym checkout has local changes"
test ! -e "$MOBILEGYM_DIR/mobilegym-data" || { echo "MobileGym's optional dataset directory is present; this study ran without it"; exit 1; }

cd "$MOBILEGYM_DIR"
echo "node $(node --version), npm $(npm --version)"
npm ci --ignore-scripts --no-audit --no-fund
NODE_OPTIONS=--max-old-space-size=4096 npm run build
"${SIM_PY:-python}" -m playwright install chromium

node "$REPO_DIR/textview/build_icon_table.mjs" "$MOBILEGYM_DIR/node_modules/lucide-react/dist/esm/icons" "$REPO_DIR/textview/lucide_icon_table.json"
got=$("${SIM_PY:-python}" -c "import hashlib,sys; print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$REPO_DIR/textview/lucide_icon_table.json")
[ "$got" = "$ICONS_MD5" ] && echo "icon table matches the study's ($got)" || echo "warning: icon table md5 $got differs from the study's $ICONS_MD5"
echo SETUP_DONE
