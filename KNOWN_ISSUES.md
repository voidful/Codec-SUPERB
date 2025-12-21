# Known Issues

## LLVM SVML Symbol Error

### Problem

```
LLVM ERROR: Symbol not found: __svml_cosf8_ha
```

This error occurs on some systems with Intel MKL during metrics computation.

### System-Level Solutions

#### Option 1: Set Environment Variables (Recommended)

```bash
export MKL_SERVICE_FORCE_INTEL=1
export KMP_DUPLICATE_LIB_OK=TRUE
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

python3 benchmarking.py --dataset <path> --models <codec> --max_workers 1
```

#### Option 2: Reinstall PyTorch with CPU-only Build

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### Option 3: Use Conda with MKL-free NumPy

```bash
conda install nomkl numpy scipy
```

### Why Not Fixed in Code?

This is a system-level library compatibility issue that should be resolved in the user's environment, not worked around in application code. Different systems may require different solutions.
