#!/bin/bash
# Quick fix script for LLVM SVML errors
# Run this instead of directly running python

MKL_THREADING_LAYER=GNU \
MKL_SERVICE_FORCE_INTEL=1 \
KMP_DUPLICATE_LIB_OK=TRUE \
OMP_NUM_THREADS=1 \
TF_ENABLE_ONEDNN_OPTS=0 \
PYTHONPATH=. \
python3 scripts/benchmarking.py "$@"
