#!/bin/bash
# Wrapper script to run benchmarking.py with proper environment variables
# This fixes LLVM SVML symbol errors

export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Run the benchmarking script with all arguments
PYTHONPATH=. python3 scripts/benchmarking.py "$@"
