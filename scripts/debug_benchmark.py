#!/usr/bin/env python3
"""
Debug script to check what's causing the hang in benchmarking
"""

import sys
import time
from datasets import load_from_disk

print("=" * 60)
print("DEBUG: Checking dataset loading")
print("=" * 60)

dataset_path = "datasets/voidful/codec-superb-tiny_synth"

print(f"\n1. Loading dataset from: {dataset_path}")
start = time.time()
try:
    c = load_from_disk(dataset_path)
    print(f"   ✅ Loaded in {time.time() - start:.2f}s")
    print(f"   Splits: {list(c.keys())}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print(f"\n2. Checking 'llmcodec' split exists")
if 'llmcodec' in c:
    print(f"   ✅ Found llmcodec split")
    print(f"   Size: {len(c['llmcodec'])} samples")
else:
    print(f"   ❌ llmcodec split not found!")
    print(f"   Available: {list(c.keys())}")
    sys.exit(1)

print(f"\n3. Checking 'original' split exists")
if 'original' in c:
    print(f"   ✅ Found original split")
    print(f"   Size: {len(c['original'])} samples")
else:
    print(f"   ❌ original split not found!")
    sys.exit(1)

print(f"\n4. Loading first sample from llmcodec")
start = time.time()
try:
    sample = c['llmcodec'][0]
    print(f"   ✅ Loaded in {time.time() - start:.2f}s")
    print(f"   Keys: {list(sample.keys())}")
    
    if 'audio' in sample:
        audio = sample['audio']
        print(f"   Audio type: {type(audio)}")
        if isinstance(audio, dict):
            print(f"   Audio keys: {list(audio.keys())}")
            if 'array' in audio:
                arr = audio['array']
                print(f"   Array type: {type(arr)}")
                print(f"   Array shape: {arr.shape if hasattr(arr, 'shape') else 'N/A'}")
                
                # Check if it's a CUDA tensor
                if hasattr(arr, 'is_cuda'):
                    print(f"   ⚠️  Is CUDA tensor: {arr.is_cuda}")
                if hasattr(arr, 'device'):
                    print(f"   ⚠️  Device: {arr.device}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n5. Loading first sample from original")
start = time.time()
try:
    sample = c['original'][0]
    print(f"   ✅ Loaded in {time.time() - start:.2f}s")
    
    if 'audio' in sample:
        audio = sample['audio']
        if isinstance(audio, dict) and 'array' in audio:
            arr = audio['array']
            print(f"   Array type: {type(arr)}")
            if hasattr(arr, 'is_cuda'):
                print(f"   ⚠️  Is CUDA tensor: {arr.is_cuda}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print(f"\n6. Testing iteration (first 3 samples)")
start = time.time()
try:
    for i, (orig, model) in enumerate(zip(c['original'], c['llmcodec'])):
        if i >= 3:
            break
        print(f"   Sample {i}: ", end='', flush=True)
        
        # Try to access the audio
        orig_audio = orig['audio']
        model_audio = model['audio']
        
        print(f"✅ (took {time.time() - start:.2f}s)")
        start = time.time()
        
except Exception as e:
    print(f"\n   ❌ Failed at sample {i}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n7. Testing metrics computation on first sample")
start = time.time()
try:
    from benchmarking import compute_metrics
    
    orig = c['original'][0]
    model = c['llmcodec'][0]
    
    print(f"   Computing metrics...", flush=True)
    metrics = compute_metrics(orig, model, max_duration=120)
    print(f"   ✅ Computed in {time.time() - start:.2f}s")
    print(f"   Metrics: {metrics}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED")
print("=" * 60)
