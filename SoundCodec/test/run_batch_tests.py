#!/usr/bin/env python3
"""
Simple test runner for batch processing functionality in Codec-SUPERB.

This script provides an easy way to run all batch processing tests and benchmarks.
"""

import os
import sys
import subprocess
import argparse


def run_command(command: str, description: str):
    """Run a command and display the results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run batch processing tests and benchmarks")
    parser.add_argument('--tests-only', action='store_true',
                       help='Run only unit tests, skip benchmarks')
    parser.add_argument('--benchmarks-only', action='store_true',
                       help='Run only benchmarks, skip unit tests')
    parser.add_argument('--quick-benchmark', action='store_true',
                       help='Run quick benchmark with smaller batch sizes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("Codec-SUPERB Batch Processing Test Suite")
    print("=" * 45)
    
    # Get the test directory path
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    success_count = 0
    total_count = 0
    
    # Run unit tests unless benchmarks-only is specified
    if not args.benchmarks_only:
        print(f"\n🧪 RUNNING UNIT TESTS")
        print(f"{'='*30}")
        
        # Test with pytest if available, otherwise with python
        pytest_cmd = f"python -m pytest {test_dir}/test_batch_processing.py"
        if args.verbose:
            pytest_cmd += " -v"
        
        total_count += 1
        if run_command(pytest_cmd, "Unit Tests"):
            success_count += 1
    
    # Run benchmarks unless tests-only is specified
    if not args.tests_only:
        print(f"\n🚀 RUNNING PERFORMANCE BENCHMARKS")
        print(f"{'='*35}")
        
        benchmark_cmd = f"python {test_dir}/benchmark_batch_performance.py"
        
        if args.quick_benchmark:
            benchmark_cmd += " --batch-sizes 1 2 4 --duration 0.5"
        else:
            benchmark_cmd += " --batch-sizes 1 2 4 8"
        
        if args.verbose:
            benchmark_cmd += " --memory-profile"
        
        total_count += 1
        if run_command(benchmark_cmd, "Performance Benchmarks"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests completed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        exit_code = 0
    else:
        print("❌ Some tests failed!")
        exit_code = 1
    
    print(f"\n📋 Next Steps:")
    if not args.tests_only:
        print("• Check benchmark results for performance improvements")
        print("• Consider adjusting batch sizes based on your hardware")
    
    print("• Use batch processing in your code for better performance:")
    print("  batch_extracted = codec.batch_extract_unit(data_list)")
    print("  batch_decoded = codec.batch_decode_unit(batch_extracted)")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 