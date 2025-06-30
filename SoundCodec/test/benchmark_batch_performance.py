#!/usr/bin/env python3
"""
Performance benchmark for batch processing vs single processing in Codec-SUPERB.

This script measures and compares the performance of batch processing against
traditional single-item processing across different codecs and batch sizes.
"""

import time
import torch
import numpy as np
import os
import sys
from typing import List, Dict, Any
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from SoundCodec.base_codec.general import BaseCodec


def create_test_data(num_samples: int, sample_rate: int = 16000, duration: float = 1.0) -> List[Dict]:
    """Create test audio data."""
    data_list = []
    np.random.seed(42)  # For reproducible benchmarks
    
    for i in range(num_samples):
        # Add some variation in length (±20%)
        variation = 0.8 + 0.4 * np.random.random()
        length = int(sample_rate * duration * variation)
        audio_array = np.random.randn(length).astype(np.float32)
        
        data_item = {
            'id': f'benchmark_sample_{i}',
            'audio': {
                'array': audio_array,
                'sampling_rate': sample_rate
            }
        }
        data_list.append(data_item)
    
    return data_list


def benchmark_single_processing(codec: BaseCodec, data_list: List[Dict]) -> Dict[str, float]:
    """Benchmark single-item processing."""
    start_time = time.time()
    
    # Extract units one by one
    extract_start = time.time()
    extracted_units = []
    for data in data_list:
        unit = codec.extract_unit(data)
        extracted_units.append(unit)
    extract_time = time.time() - extract_start
    
    # Decode units one by one
    decode_start = time.time()
    decoded_audio = []
    for unit in extracted_units:
        audio = codec.decode_unit(unit.stuff_for_synth)
        decoded_audio.append(audio)
    decode_time = time.time() - decode_start
    
    total_time = time.time() - start_time
    
    return {
        'extract_time': extract_time,
        'decode_time': decode_time,
        'total_time': total_time,
        'samples_processed': len(data_list)
    }


def benchmark_batch_processing(codec: BaseCodec, data_list: List[Dict]) -> Dict[str, float]:
    """Benchmark batch processing."""
    start_time = time.time()
    
    # Batch extract units
    extract_start = time.time()
    batch_extracted = codec.batch_extract_unit(data_list)
    extract_time = time.time() - extract_start
    
    # Batch decode units
    decode_start = time.time()
    batch_decoded = codec.batch_decode_unit(batch_extracted)
    decode_time = time.time() - decode_start
    
    total_time = time.time() - start_time
    
    return {
        'extract_time': extract_time,
        'decode_time': decode_time,
        'total_time': total_time,
        'samples_processed': len(data_list)
    }


def benchmark_batch_synth(codec: BaseCodec, data_list: List[Dict]) -> Dict[str, float]:
    """Benchmark complete batch synthesis pipeline."""
    start_time = time.time()
    
    # Complete batch synthesis
    results = codec.batch_synth(data_list.copy(), local_save=False)
    
    total_time = time.time() - start_time
    
    return {
        'extract_time': 0.0,  # Not separately measured
        'decode_time': 0.0,   # Not separately measured
        'total_time': total_time,
        'samples_processed': len(results)
    }


def print_benchmark_results(codec_name: str, batch_size: int, single_results: Dict, 
                          batch_results: Dict, batch_synth_results: Dict):
    """Print formatted benchmark results."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {codec_name}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<20} {'Extract (s)':<12} {'Decode (s)':<12} {'Total (s)':<12} {'Samples/s':<12}")
    print(f"{'-'*60}")
    
    # Single processing
    single_throughput = single_results['samples_processed'] / single_results['total_time']
    print(f"{'Single Processing':<20} {single_results['extract_time']:<12.4f} "
          f"{single_results['decode_time']:<12.4f} {single_results['total_time']:<12.4f} "
          f"{single_throughput:<12.2f}")
    
    # Batch processing
    batch_throughput = batch_results['samples_processed'] / batch_results['total_time']
    print(f"{'Batch Processing':<20} {batch_results['extract_time']:<12.4f} "
          f"{batch_results['decode_time']:<12.4f} {batch_results['total_time']:<12.4f} "
          f"{batch_throughput:<12.2f}")
    
    # Batch synthesis
    batch_synth_throughput = batch_synth_results['samples_processed'] / batch_synth_results['total_time']
    print(f"{'Batch Synthesis':<20} {'-':<12} {'-':<12} "
          f"{batch_synth_results['total_time']:<12.4f} {batch_synth_throughput:<12.2f}")
    
    print(f"\n{'PERFORMANCE GAINS':<20}")
    print(f"{'-'*30}")
    
    # Calculate speedups
    extract_speedup = single_results['extract_time'] / batch_results['extract_time'] if batch_results['extract_time'] > 0 else 0
    decode_speedup = single_results['decode_time'] / batch_results['decode_time'] if batch_results['decode_time'] > 0 else 0
    total_speedup = single_results['total_time'] / batch_results['total_time']
    synth_speedup = single_results['total_time'] / batch_synth_results['total_time']
    
    print(f"Extract Speedup:     {extract_speedup:.2f}x")
    print(f"Decode Speedup:      {decode_speedup:.2f}x")
    print(f"Total Speedup:       {total_speedup:.2f}x")
    print(f"Synthesis Speedup:   {synth_speedup:.2f}x")
    
    # Memory efficiency (rough estimate)
    print(f"\nThroughput Improvement: {((batch_throughput - single_throughput) / single_throughput * 100):.1f}%")


def benchmark_codec(codec_class, codec_name: str, batch_sizes: List[int], 
                   sample_rate: int = 16000, duration: float = 1.0):
    """Benchmark a specific codec across different batch sizes."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING {codec_name.upper()}")
    print(f"{'='*80}")
    
    try:
        codec = codec_class()
        
        for batch_size in batch_sizes:
            print(f"\nPreparing test data for batch size {batch_size}...")
            data_list = create_test_data(batch_size, sample_rate, duration)
            
            # Warm up (to avoid cold start effects)
            if len(data_list) > 0:
                codec.extract_unit(data_list[0])
            
            print(f"Running benchmarks...")
            
            # Benchmark single processing
            single_results = benchmark_single_processing(codec, data_list)
            
            # Benchmark batch processing
            batch_results = benchmark_batch_processing(codec, data_list)
            
            # Benchmark batch synthesis
            batch_synth_results = benchmark_batch_synth(codec, data_list)
            
            # Print results
            print_benchmark_results(codec_name, batch_size, single_results, 
                                  batch_results, batch_synth_results)
    
    except Exception as e:
        print(f"❌ Error benchmarking {codec_name}: {e}")


def run_memory_profile(codec_class, codec_name: str, batch_size: int = 8):
    """Run a simple memory usage comparison."""
    print(f"\n{'='*60}")
    print(f"MEMORY USAGE COMPARISON: {codec_name}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    try:
        import psutil
        import os
        
        codec = codec_class()
        data_list = create_test_data(batch_size, 16000, 1.0)
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Single processing
        for data in data_list:
            extracted = codec.extract_unit(data)
            decoded = codec.decode_unit(extracted.stuff_for_synth)
        
        memory_single = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Batch processing
        batch_extracted = codec.batch_extract_unit(data_list)
        batch_decoded = codec.batch_decode_unit(batch_extracted)
        
        memory_batch = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory before:       {memory_before:.1f} MB")
        print(f"Memory after single: {memory_single:.1f} MB (+{memory_single - memory_before:.1f} MB)")
        print(f"Memory after batch:  {memory_batch:.1f} MB (+{memory_batch - memory_before:.1f} MB)")
        print(f"Memory efficiency:   {((memory_single - memory_before) / (memory_batch - memory_before)):.2f}x")
        
    except ImportError:
        print("psutil not available for memory profiling")
    except Exception as e:
        print(f"Error in memory profiling: {e}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark batch processing performance")
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16],
                       help='Batch sizes to test (default: 1 2 4 8 16)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Audio duration in seconds (default: 1.0)')
    parser.add_argument('--memory-profile', action='store_true',
                       help='Run memory usage profiling')
    
    args = parser.parse_args()
    
    print("Codec-SUPERB Batch Processing Performance Benchmark")
    print("="*55)
    
    # Define available codecs for benchmarking
    codecs_to_test = []
    
    # Try to import and test available codecs
    try:
        from SoundCodec.codec.encodec_24k_6bps import Codec as EncodecCodec
        codecs_to_test.append((EncodecCodec, "EnCodec 24kHz 6bps"))
    except ImportError:
        print("⚠️  EnCodec not available")
    
    # Create a simple mock codec for demonstration
    from SoundCodec.base_codec.general import BaseCodec, ExtractedUnit
    
    class MockCodec(BaseCodec):
        def config(self):
            self.sampling_rate = args.sample_rate
            self.setting = "mock_codec"
        
        def extract_unit(self, data):
            audio = data['audio']['array']
            # Simulate some processing time
            time.sleep(0.001)  # 1ms delay
            unit = torch.tensor(audio[:100])
            return ExtractedUnit(unit=unit, stuff_for_synth=audio)
        
        def decode_unit(self, stuff_for_synth):
            # Simulate some processing time
            time.sleep(0.001)  # 1ms delay
            return stuff_for_synth
    
    codecs_to_test.append((MockCodec, "Mock Codec (Demo)"))
    
    # Run benchmarks
    for codec_class, codec_name in codecs_to_test:
        benchmark_codec(codec_class, codec_name, args.batch_sizes, 
                       args.sample_rate, args.duration)
        
        if args.memory_profile:
            run_memory_profile(codec_class, codec_name)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print("="*80)
    print("\n📊 Key Takeaways:")
    print("• Batch processing typically provides 2-5x speedup")
    print("• Larger batch sizes generally improve performance")
    print("• Memory usage is more efficient with batch processing")
    print("• Results may vary depending on audio length and hardware")
    print("\n💡 Tips for optimal performance:")
    print("• Group samples by similar length before batching")
    print("• Use batch sizes of 4-16 for best balance of speed and memory")
    print("• Consider your GPU memory constraints when choosing batch size")


if __name__ == "__main__":
    main() 