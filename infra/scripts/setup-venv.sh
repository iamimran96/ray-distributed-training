#!/usr/bin/env bash
# Create .venv in the repo root and install the Ray CLI from requirements.txt.
# Works in Git Bash on Windows and on Linux/macOS.
#   ./infra/scripts/setup-venv.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${REPO_DIR}/.venv"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv at .venv with $("$PYTHON" --version)"
  "$PYTHON" -m venv "$VENV"
fi

if [[ -x "$VENV/Scripts/python.exe" ]]; then
  VPY="$VENV/Scripts/python.exe"; ACTIVATE=".venv/Scripts/activate"
else
  VPY="$VENV/bin/python"; ACTIVATE=".venv/bin/activate"
fi

echo "==> Installing requirements"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -r "${REPO_DIR}/requirements.txt"
"$VPY" -c "import ray; print('ray', ray.__version__)"

cat <<EOF

Activate it with:
  source ${ACTIVATE}                  # Git Bash / Linux / macOS
  .venv\\Scripts\\Activate.ps1          # PowerShell
Then:
  export RAY_ADDRESS=http://127.0.0.1:8265   # PowerShell: \$env:RAY_ADDRESS="http://127.0.0.1:8265"
  ray job list
EOF
