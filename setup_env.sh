
#!/usr/bin/env bash
#
# Usage:
#   bash setup_env.sh
#   conda activate crosscoder

set -euo pipefail

ENV_NAME="crosscoder"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found in PATH. Install Miniconda or Anaconda first." >&2
    exit 1
fi

# Enable `conda activate` inside a non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$ENV_NAME"; then
    echo "Env '$ENV_NAME' already exists — updating packages."
    conda activate "$ENV_NAME"
else
    echo "Creating env '$ENV_NAME' with Python $PYTHON_VERSION."
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    conda activate "$ENV_NAME"
fi

pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Done. To use it in your current shell:"
echo "  conda activate $ENV_NAME"
