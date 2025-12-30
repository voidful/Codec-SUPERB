#!/bin/bash
# Wrapper script to run benchmarking.py with proper environment variables
# This fixes LLVM SVML symbol errors

# Set environment variables to fix LLVM/MKL issues
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Run the benchmarking script with all arguments
PYTHONPATH=. python3 scripts/benchmarking.py "$@"
