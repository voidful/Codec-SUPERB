# Fixing LLVM SVML Symbol Error

If you encounter `LLVM ERROR: Symbol not found: __svml_cosf8_ha`, this is caused by Intel MKL library conflicts.

## Quick Fix (Recommended)

Use the wrapper script:

```bash
./scripts/run_benchmark.sh --dataset voidful/codec-superb-tiny --models auv,llmcodec,bigcodec_1k
```

## Manual Fix Options

### Option 1: Environment Variables (Before Running Python)

```bash
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1  
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

PYTHONPATH=. python3 scripts/benchmarking.py --dataset voidful/codec-superb-tiny --models auv,llmcodec,bigcodec_1k
```

### Option 2: Reinstall NumPy without MKL

```bash
pip uninstall numpy
pip install numpy --no-binary numpy
```

### Option 3: Use OpenBLAS Instead

```bash
pip uninstall numpy scipy
pip install numpy scipy --no-binary :all:
```

### Option 4: Downgrade NumPy

```bash
pip install numpy==1.24.3 --force-reinstall
```

### Option 5: Install Intel MKL Properly (If Using Conda)

```bash
conda install -c intel mkl mkl-service
```

## Root Cause

This error occurs when:

- PyTorch/NumPy is compiled with Intel MKL
- The MKL library is missing or incompatible
- SVML (Short Vector Math Library) symbols cannot be found

## Testing the Fix

After applying any fix, test with:

```bash
python3 -c "import numpy as np; print(np.__version__); print(np.show_config())"
```
