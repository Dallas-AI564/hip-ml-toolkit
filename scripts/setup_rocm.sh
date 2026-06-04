#!/bin/bash
# setup_rocm.sh — Set up ROCm environment for hip-ml-toolkit

set -e

echo "=== hip-ml-toolkit ROCm Setup ==="

# Check ROCm
if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: ROCm not found."
    echo "Install: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

echo "ROCm version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"

# Check GPU
echo ""
echo "=== Detected AMD GPUs ==="
rocm-smi --showproductname 2>/dev/null || hip-smi 2>/dev/null || echo "No GPU detected"

# Check Python
echo ""
echo "=== Python ==="
python3 --version

# Install package
echo ""
echo "=== Installing hip-ml-toolkit ==="
pip install -e ".[dev,serve]"

echo ""
echo "=== Setup complete ==="
echo "Run: hipml info"
