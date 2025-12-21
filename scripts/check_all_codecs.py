import torch
import torchaudio
import numpy as np
import os
import sys
import psutil
import time
from SoundCodec.codec import list_codec, load_codec
import traceback

def check_all_codecs():
    print(f"Starting codec verification on {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB RAM")
    
    # Use a sample file from the test directory
    sample_file = "SoundCodec/test/sample1_16k.wav"
    if not os.path.exists(sample_file):
        print(f"Error: Sample file {sample_file} not found.")
        return

    # Load sample audio
    try:
        import soundfile as sf
        waveform_np, sample_rate = sf.read(sample_file)
        # waveform_np can be (time,) or (time, channels)
        if len(waveform_np.shape) > 1:
            # use last channel as in original logic
            waveform_last_channel = waveform_np[:, -1]
        else:
            waveform_last_channel = waveform_np
        
        waveform_np_to_use = waveform_last_channel
    except Exception as e:
        print(f"Loading failed: {e}")
        return

    codecs = list_codec()
    print(f"Found {len(codecs)} codecs to test: {', '.join(codecs)}")
    
    results = []
    
    for codec_name in codecs:
        print(f"\n[{codec_name}] Testing...")
        start_time = time.time()
        res = {
            'name': codec_name,
            'synth_status': '❓',
            'unit_shape': '-',
            'effective_ndim': '-',
            'is_1d': '-',
            'time': '-',
            'error': ''
        }
        
        try:
            # Load codec
            codec = load_codec(codec_name)
            
            # 1. Check Dimensions (using extract_unit with dummy data)
            dummy_data = {
                "audio": {
                    "array": np.zeros(16000),
                    "sampling_rate": 16000
                }
            }
            try:
                extracted = codec.extract_unit(dummy_data)
                unit = extracted.unit
                res['unit_shape'] = str(list(unit.shape))
                
                # Check effective ndim manually for reporting
                if hasattr(unit, 'squeeze'):
                     squeezed_unit = unit.squeeze()
                     dims = squeezed_unit.shape if hasattr(squeezed_unit, 'shape') else ()
                else:
                     dims = unit.shape

                # effective_dims = [d for d in dims if d > 1] # This logic depends on exact shape
                # Easier: use the codec's is_1d method if available, or manual check
                
                is_1d_status = codec.is_1d()
                res['is_1d'] = str(is_1d_status)
                
                # Re-calculate effective ndim for display (similar to check_all_codecs_dim.py)
                # We use the raw unit shape filtering for consistency
                raw_dims = unit.shape
                effective_dims_list = [d for d in raw_dims if d > 1]
                res['effective_ndim'] = str(len(effective_dims_list))

            except Exception as e:
                res['unit_shape'] = 'Error'
                print(f"  Dimension check failed: {e}")

            # 2. Check Synthesis (using real sample)
            data_item = {
                'id': 'test_sample',
                'audio': {
                    'array': waveform_np_to_use,
                    'sampling_rate': sample_rate
                }
            }
            
            output = codec.synth(data_item, local_save=False)
            
            if 'audio' in output and 'array' in output['audio']:
                output_array = output['audio']['array']
                duration = time.time() - start_time
                print(f"  Success! (Time: {duration:.2f}s, Output: {output_array.shape})")
                res['synth_status'] = '✅ Pass'
                res['time'] = f"{duration:.2f}s"
            else:
                print(f"  Failed: 'audio.array' not in output")
                res['synth_status'] = '❌ Fail'
                res['error'] = "Missing audio output"
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e).split('\n')[0]
            print(f"  Failed: {error_msg}")
            res['synth_status'] = '❌ Fail'
            res['time'] = f"{duration:.2f}s"
            res['error'] = error_msg
        
        results.append(res)
        
        # Cleanup
        if 'codec' in locals():
            del codec
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary table
    print("\n" + "="*120)
    print(f"{'Codec Name':<40} | {'Synth':<6} | {'Shape':<20} | {'Ndim':<5} | {'1D?':<6} | {'Time':<8} | {'Error'}")
    print("-" * 120)
    for res in results:
        print(f"{res['name']:<40} | {res['synth_status']:<6} | {res['unit_shape']:<20} | {res['effective_ndim']:<5} | {res['is_1d']:<6} | {res['time']:<8} | {res['error']}")
    print("="*120)

if __name__ == "__main__":
    check_all_codecs()
