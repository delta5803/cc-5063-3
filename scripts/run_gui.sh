#!/usr/bin/env bash
# Launch the graphical surface fairing editor (primary interface).
# Requires the project virtual environment created during setup.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d "scratch/sysroot" ]; then
  export LD_LIBRARY_PATH="$PWD/scratch/sysroot/usr/lib/aarch64-linux-gnu:$PWD/scratch/sysroot/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv/bin/python -m surface_fairing.gui
