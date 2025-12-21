#!/bin/bash
# Wrapper script to run benchmarking with proper environment variables
# to avoid LLVM SVML symbol errors

# Set environment variables before Python starts
export MKL_SERVICE_FORCE_INTEL=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run benchmarking with all arguments passed through
python3 benchmarking.py "$@"
