import torch
import torchaudio
import numpy as np
import os
import sys
import psutil
import time
from SoundCodec.codec import list_codec, load_codec
from collections import defaultdict
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
        try:
            # Load codec
            codec = load_codec(codec_name)
            
            # Prepare data item
            # Most codecs expect a specific sampling rate, but synth usually handles resampling or expects it
            # We'll provide the original sample rate and array
            data_item = {
                'id': 'test_sample',
                'audio': {
                    'array': waveform_np_to_use,
                    'sampling_rate': sample_rate
                }
            }
            
            # Run synth (inference)
            # local_save=False to avoid cluttering with wav files
            output = codec.synth(data_item, local_save=False)
            
            # Check output
            if 'audio' in output and 'array' in output['audio']:
                output_array = output['audio']['array']
                duration = time.time() - start_time
                print(f"[{codec_name}] Success! (Time: {duration:.2f}s, Output shape: {output_array.shape})")
                results.append({
                    'name': codec_name,
                    'status': '✅ Pass',
                    'time': f"{duration:.2f}s",
                    'error': ''
                })
            else:
                print(f"[{codec_name}] Failed: 'audio.array' not in output")
                results.append({
                    'name': codec_name,
                    'status': '❌ Fail',
                    'time': '-',
                    'error': "'audio.array' missing in output"
                })
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e).split('\n')[0] # Get first line of error
            print(f"[{codec_name}] Failed: {error_msg}")
            # traceback.print_exc()
            results.append({
                'name': codec_name,
                'status': '❌ Fail',
                'time': f"{duration:.2f}s",
                'error': error_msg
            })
        
        # Cleanup
        if 'codec' in locals():
            del codec
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # gc.collect() # Optional

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Codec Name':<45} | {'Status':<10} | {'Time':<10} | {'Error'}")
    print("-" * 80)
    for res in results:
        print(f"{res['name']:<45} | {res['status']:<10} | {res['time']:<10} | {res['error']}")
    print("="*80)

if __name__ == "__main__":
    check_all_codecs()
