#!/usr/bin/env bash
# Setup script for TabbyAPI (ExLlamaV2-based inference server)
# Run once before first use.
#
# Usage: bash setup_tabbyapi.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABBY_DIR="$SCRIPT_DIR/tabbyAPI"

echo "=== TabbyAPI Setup ==="

# 1. Clone TabbyAPI if not present
if [ ! -d "$TABBY_DIR" ]; then
    echo "[1/4] Cloning TabbyAPI..."
    git clone https://github.com/theroyallab/tabbyAPI.git "$TABBY_DIR"
else
    echo "[1/4] TabbyAPI already cloned at $TABBY_DIR"
fi

cd "$TABBY_DIR"

# 2. Install Python dependencies
echo "[2/4] Installing TabbyAPI dependencies..."
pip3 install -r requirements.txt

# 3. Install ExLlamaV2 (CUDA build from PyPI)
echo "[3/4] Installing ExLlamaV2..."
pip3 install exllamav2

# 4. Install project dependencies
echo "[4/4] Installing project dependencies..."
cd "$SCRIPT_DIR"
pip3 install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Download models to ~/models/ (see TASKS.md for model names)"
echo "  2. Start TabbyAPI:  cd tabbyAPI && python3 main.py --port 5000"
echo "  3. Run a debate:    python3 main.py --check"
echo "                      python3 main.py --topic 'Your debate topic here'"
echo ""
echo "TabbyAPI config file: tabbyAPI/config.yml"
echo "Edit it to set model directory and auth key if needed."
