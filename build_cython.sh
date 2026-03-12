#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────── #
# Cython Build Script for GRAVITAS Engine                                      #
# ─────────────────────────────────────────────────────────────────────────── #
#
# Compiles performance-critical numerical kernels to native machine code
# using Cython + GCC with aggressive optimizations (-O3, -ffast-math,
# -march=native, -funroll-loops).
#
# Compared to Nuitka (which preserves Python semantics), Cython compiles
# typed numerical code directly to C with:
#   - No Python object overhead for typed variables
#   - C-level math (libc pow, sqrt, tanh, fabs)
#   - Typed memoryviews → direct memory access (no NumPy per-call overhead)
#   - Bounds-check elimination, loop unrolling
#   - ~10-50x speedup on small-array numerical kernels
#
# Prerequisites:
#   pip install cython>=3.0 numpy
#   apt install gcc   # (or any C compiler)
#
# Usage:
#   ./build_cython.sh              # Compile Cython extensions in-place
#   ./build_cython.sh --clean      # Remove build artifacts
#   ./build_cython.sh --annotate   # Compile + generate HTML annotation
#
# ─────────────────────────────────────────────────────────────────────────── #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer .venv Python if it exists
if [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
else
    PYTHON="${PYTHON:-$(which python3)}"
fi

# ── Clean mode ────────────────────────────────────────────────────────────── #
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning Cython build artifacts..."
    rm -rf build/ dist/ *.egg-info
    find . -name '*.so' -path '*/core/*' -delete
    find . -name '*.c' -path '*/core/_kernels.c' -delete
    find . -name '*.html' -path '*/core/_kernels.html' -delete
    find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
    echo "Done."
    exit 0
fi

# ── Version info ──────────────────────────────────────────────────────────── #
echo "═══════════════════════════════════════════════════════════"
echo "  GRAVITAS Engine — Cython Build"
echo "  Python:  $($PYTHON --version 2>&1)"
echo "  Cython:  $($PYTHON -c 'import Cython; print(Cython.__version__)' 2>&1)"
echo "  NumPy:   $($PYTHON -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "  CC:      ${CC:-$(which gcc || which cc)}"
echo "═══════════════════════════════════════════════════════════"
echo

# ── Build ─────────────────────────────────────────────────────────────────── #
echo "[1/1] Compiling Cython kernels (gravitas_engine.core._kernels)..."
echo "      Optimizations: -O3 -march=native -funroll-loops -fno-math-errno"
echo

$PYTHON setup.py build_ext --inplace 2>&1

echo
echo "═══════════════════════════════════════════════════════════"
echo "  Build complete. Compiled extensions:"
find . -name '*.so' -path '*/core/*' -exec ls -lh {} \;
echo "═══════════════════════════════════════════════════════════"
echo
echo "No PYTHONPATH changes needed — extensions are built in-place."
echo "Run training as usual:"
echo "  python tests/train_moscow_selfplay.py"
echo "  python cli.py run moscow --episodes 30"
