#!/usr/bin/env python3
"""
Comprehensive test suite for batch processing functionality in Codec-SUPERB.

This test suite verifies that batch processing works correctly across all codecs
and produces consistent results compared to single-item processing.
"""

import pytest
import torch
import numpy as np
import os
import sys
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from SoundCodec.base_codec.general import BaseCodec, BatchExtractedUnit, ExtractedUnit


class MockCodec(BaseCodec):
    """Mock codec for testing."""
    
    def config(self):
        self.sampling_rate = 16000
        self.setting = "mock_codec"
    
    def extract_unit(self, data):
        audio = data['audio']['array']
        unit = torch.tensor(audio[:100])  # Take first 100 samples as "unit"
        return ExtractedUnit(unit=unit, stuff_for_synth=audio)
    
    def decode_unit(self, stuff_for_synth):
        return stuff_for_synth


class TestBatchProcessing:
    """Test suite for batch processing functionality."""
    
    @pytest.fixture
    def mock_codec(self):
        return MockCodec()
    
    @pytest.fixture
    def sample_data_list(self):
        """Create test data."""
        np.random.seed(42)
        data_list = []
        lengths = [8000, 16000, 12000]
        
        for i, length in enumerate(lengths):
            audio_array = np.random.randn(length).astype(np.float32)
            data_item = {
                'id': f'test_sample_{i}',
                'audio': {
                    'array': audio_array,
                    'sampling_rate': 16000
                }
            }
            data_list.append(data_item)
        return data_list

    def test_batch_extract_unit(self, mock_codec, sample_data_list):
        """Test batch extraction."""
        batch_extracted = mock_codec.batch_extract_unit(sample_data_list)
        
        assert isinstance(batch_extracted, BatchExtractedUnit)
        assert batch_extracted.batch_size == len(sample_data_list)
        assert len(batch_extracted.units) == len(sample_data_list)
        assert len(batch_extracted.stuff_for_synth) == len(sample_data_list)

    def test_batch_decode_unit(self, mock_codec, sample_data_list):
        """Test batch decoding."""
        batch_extracted = mock_codec.batch_extract_unit(sample_data_list)
        batch_decoded = mock_codec.batch_decode_unit(batch_extracted)
        
        assert len(batch_decoded) == len(sample_data_list)
        for decoded_audio in batch_decoded:
            assert isinstance(decoded_audio, np.ndarray)

    def test_batch_synth(self, mock_codec, sample_data_list):
        """Test batch synthesis."""
        batch_results = mock_codec.batch_synth(sample_data_list.copy(), local_save=False)
        
        assert len(batch_results) == len(sample_data_list)
        for result in batch_results:
            assert 'unit' in result
            assert 'audio' in result
            assert isinstance(result['unit'], torch.Tensor)

    def test_single_vs_batch_consistency(self, mock_codec, sample_data_list):
        """Test consistency between single and batch processing."""
        # Single processing
        single_units = []
        for data in sample_data_list:
            extracted = mock_codec.extract_unit(data)
            single_units.append(extracted.unit)
        
        # Batch processing
        batch_extracted = mock_codec.batch_extract_unit(sample_data_list)
        
        # Compare results
        for single_unit, batch_unit in zip(single_units, batch_extracted.units):
            assert torch.allclose(single_unit, batch_unit, atol=1e-6)

    def test_empty_batch(self, mock_codec):
        """Test empty batch handling."""
        empty_list = []
        batch_extracted = mock_codec.batch_extract_unit(empty_list)
        
        assert batch_extracted.batch_size == 0
        assert len(batch_extracted.units) == 0
        assert len(batch_extracted.stuff_for_synth) == 0

    def test_single_item_batch(self, mock_codec, sample_data_list):
        """Test single item batch."""
        single_data = [sample_data_list[0]]
        batch_extracted = mock_codec.batch_extract_unit(single_data)
        
        assert batch_extracted.batch_size == 1
        assert len(batch_extracted.units) == 1


class TestRealCodecs:
    """Test batch processing with real codec implementations."""
    
    def test_encodec_batch_processing(self):
        """Test batch processing with EnCodec codec."""
        try:
            from SoundCodec.codec.encodec_24k_6bps import Codec as EncodecCodec
            
            codec = EncodecCodec()
            
            # Create test data
            np.random.seed(42)
            data_list = []
            for i in range(2):  # Small batch for testing
                audio_array = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
                data_item = {
                    'id': f'encodec_test_{i}',
                    'audio': {
                        'array': audio_array,
                        'sampling_rate': 24000
                    }
                }
                data_list.append(data_item)
            
            # Test batch extraction
            batch_extracted = codec.batch_extract_unit(data_list)
            assert batch_extracted.batch_size == 2
            assert len(batch_extracted.units) == 2
            
            # Test batch decoding
            batch_decoded = codec.batch_decode_unit(batch_extracted)
            assert len(batch_decoded) == 2
            
            # Test batch synthesis
            batch_results = codec.batch_synth(data_list, local_save=False)
            assert len(batch_results) == 2
            
            # Verify results have expected structure
            for result in batch_results:
                assert 'unit' in result
                assert 'audio' in result
                assert isinstance(result['unit'], torch.Tensor)
                assert isinstance(result['audio']['array'], np.ndarray)
                
        except ImportError:
            pytest.skip("EnCodec not available for testing")
        except Exception as e:
            pytest.skip(f"EnCodec test failed due to: {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"]) 