#!/usr/bin/env python3
"""
Test script to verify all 4 S3Tokenizer variants are properly registered.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_s3tokenizer_variants():
    """Test that all 4 S3Tokenizer variants are available."""
    print("=" * 60)
    print("Testing S3Tokenizer Variant Registration")
    print("=" * 60)
    
    from SoundCodec.codec import list_codec
    
    # Get all available codecs
    all_codecs = list_codec()
    
    # Expected S3Tokenizer variants
    expected_variants = [
        's3tokenizer_v1',         # V1 50hz
        's3tokenizer_v1_25hz',    # V1 25hz
        's3tokenizer_v2_25hz',    # V2 25hz
        's3tokenizer_v3_25hz',    # V3 25hz
    ]
    
    print(f"\nTotal codecs available: {len(all_codecs)}")
    print(f"\nChecking for S3Tokenizer variants...")
    
    all_found = True
    for variant in expected_variants:
        if variant in all_codecs:
            print(f"  ✅ {variant} - FOUND")
        else:
            print(f"  ❌ {variant} - NOT FOUND")
            all_found = False
    
    if all_found:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: All 4 S3Tokenizer variants are registered!")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ FAILURE: Some S3Tokenizer variants are missing!")
        print("=" * 60)
        return False


def test_leaderboard_config():
    """Test that all variants are in the leaderboard configuration."""
    print("\n" + "=" * 60)
    print("Testing Leaderboard Configuration")
    print("=" * 60)
    
    # Read the update_leaderboard.py file
    with open('update_leaderboard.py', 'r') as f:
        content = f.read()
    
    expected_entries = [
        "'s3tokenizer_v1'",
        "'s3tokenizer_v1_25hz'",
        "'s3tokenizer_v2_25hz'",
        "'s3tokenizer_v3_25hz'",
    ]
    
    all_found = True
    for entry in expected_entries:
        if entry in content:
            print(f"  ✅ {entry} - FOUND in leaderboard config")
        else:
            print(f"  ❌ {entry} - NOT FOUND in leaderboard config")
            all_found = False
    
    if all_found:
        print("\n✅ SUCCESS: All variants in leaderboard configuration!")
        return True
    else:
        print("\n❌ FAILURE: Some variants missing from leaderboard!")
        return False


if __name__ == "__main__":
    print("Testing S3Tokenizer Variant Support\n")
    
    test1_pass = test_s3tokenizer_variants()
    test2_pass = test_leaderboard_config()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED")
        print("\nAll 4 S3Tokenizer variants are properly configured:")
        print("  • s3tokenizer_v1 (50hz) - 0.5 bps, 50 tps")
        print("  • s3tokenizer_v1_25hz - 0.25 bps, 25 tps")
        print("  • s3tokenizer_v2_25hz - 0.25 bps, 25 tps")
        print("  • s3tokenizer_v3_25hz - 0.25 bps, 25 tps")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
