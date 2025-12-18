import os
import torch
import numpy as np
from SoundCodec.codec import list_codec, load_codec

def calculate_metrics():
    print(f"{'Codec':<40} | {'BPS (kbps)':<10} | {'TPS':<10}")
    print("-" * 65)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Monkeypatch to ensure no MPS is used
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    device = 'cpu'
    
    codecs = list_codec()
    
    # Create specific test inputs
    duration = 1.0 # 1 second
    
    for name in codecs:
        try:
            # Skip problematic ones for now if they crash, but try to run all
            if name in ['bigcodec_1k', 'dac_24k', 'dac_44k', 's3tokenizer_v1']:
               # We know these are problematic on this env, but let's try or skip
               # For now, let's catch exceptions
               pass

            metric_name = name
            codec = load_codec(name)
            
            # Determine sampling rate
            sr = getattr(codec, 'sampling_rate', 16000)
            if sr is None: sr = 16000
            
            # Generate 1 second of silence/noise
            audio_data = np.random.randn(int(sr * duration)).astype(np.float32)
            
            data_item = {
                'audio': {
                    'array': audio_data,
                    'sampling_rate': sr
                }
            }
            
            if hasattr(codec, 'device'):
                 try:
                     codec.device = 'cpu'
                     codec.model.to('cpu')
                 except: pass

            with torch.no_grad():
                extracted = codec.extract_unit(data_item)
                unit = extracted.unit
            
            shape = unit.shape
            
            # Special check for SQCodec layers
            layers = 1
            if 'sqcodec' in name:
                try:
                    # Try to find quantizer depth
                    if hasattr(codec.model, 'quantizer'):
                        if hasattr(codec.model.quantizer, 'n_q'):
                             layers = codec.model.quantizer.n_q
                        elif hasattr(codec.model.quantizer, 'num_quantizers'):
                             layers = codec.model.quantizer.num_quantizers
                        # Check config
                        elif hasattr(codec.sq_codec, 'config'):
                             if hasattr(codec.sq_codec.config, 'n_q'):
                                 layers = codec.sq_codec.config.n_q
                except:
                    pass
            
            # Heuristic for TPS
            if len(shape) == 1:
                frames = shape[0]
            elif len(shape) == 2:
                # If shape is [1, T], usually T is token count.
                # If SQCodec flattens T*Q into T, we divide by layers
                frames = max(shape)
            else:
                frames = max(shape)
                
            if 'sqcodec' in name and layers > 1:
                # User says: divide by layers
                tps = (frames / layers) / duration
            else:
                tps = frames / duration
                
            print(f"{metric_name:<40} | {'?':<10} | {tps:<10.2f} (Shape: {shape}, Layers: {layers})")

        except Exception as e:
            print(f"{name:<40} | ERROR      | {e}")

if __name__ == "__main__":
    calculate_metrics()
