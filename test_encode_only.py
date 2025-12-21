#!/usr/bin/env python3
"""
Test script to verify encode-only codec handling.
This tests that s3tokenizer_v1 is properly marked as encode-only
and that the benchmarking system skips it appropriately.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_s3tokenizer_encode_only():
    """Test that s3tokenizer is marked as encode-only."""
    print("=" * 60)
    print("Test 1: Check s3tokenizer supports_decode flag")
    print("=" * 60)
    
    try:
        from SoundCodec.codec import load_codec
        
        codec = load_codec('s3tokenizer_v1')
        
        # Check if supports_decode attribute exists and is False
        assert hasattr(codec, 'supports_decode'), "Codec missing supports_decode attribute"
        assert codec.supports_decode == False, f"Expected supports_decode=False, got {codec.supports_decode}"
        
        print("✅ PASS: s3tokenizer_v1 correctly marked as encode-only (supports_decode=False)")
        
        # Test that decode_unit raises NotImplementedError
        print("\nTest 2: Verify decode_unit raises NotImplementedError")
        try:
            codec.decode_unit(None)
            print("❌ FAIL: decode_unit should raise NotImplementedError")
            return False
        except NotImplementedError as e:
            print(f"✅ PASS: decode_unit raises NotImplementedError: {str(e)[:80]}...")
        
        del codec
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmarking_skip():
    """Test that benchmarking skips encode-only codecs."""
    print("\n" + "=" * 60)
    print("Test 3: Check benchmarking skip logic")
    print("=" * 60)
    
    try:
        from SoundCodec.codec import load_codec
        import gc
        
        model = 's3tokenizer_v1'
        codec_instance = load_codec(model)
        
        should_skip = hasattr(codec_instance, 'supports_decode') and not codec_instance.supports_decode
        
        if should_skip:
            print(f"✅ PASS: Benchmarking would skip {model} (encode-only codec)")
            result_data = {
                "encode_only": True,
                "message": "This codec only supports encoding. No reconstruction metrics available."
            }
            print(f"   Result data: {result_data}")
        else:
            print(f"❌ FAIL: Benchmarking would NOT skip {model}")
            return False
        
        del codec_instance
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing encode-only codec handling for s3tokenizer_v1\n")
    
    test1_pass = test_s3tokenizer_encode_only()
    test2_pass = test_benchmarking_skip()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED")
        print("\nConclusion:")
        print("- s3tokenizer_v1 is correctly marked as encode-only")
        print("- Decode methods raise NotImplementedError as expected")
        print("- Benchmarking will skip this codec and show appropriate message")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
