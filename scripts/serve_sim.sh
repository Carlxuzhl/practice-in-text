#!/usr/bin/env bash
# Serve the built simulator on loopback only. Usage: bash scripts/serve_sim.sh [port]   (default: the port of $SIM_URL)
set -euo pipefail
: "${MOBILEGYM_DIR:?source configs/paths.env first}"
PORT=${1:-$(${SIM_PY:-python3} -c "import os,urllib.parse; print(urllib.parse.urlparse(os.environ.get('SIM_URL','http://127.0.0.1:14173')).port)")}
cd "$MOBILEGYM_DIR"
export UV_THREADPOOL_SIZE=4
exec npm run preview -- --port "$PORT" --host 127.0.0.1 --strictPort
