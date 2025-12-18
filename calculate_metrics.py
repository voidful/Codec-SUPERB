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
            # standard shape is usually (1, T) or (T,)
            audio_data = np.random.randn(int(sr * duration)).astype(np.float32)
            
            data_item = {
                'audio': {
                    'array': audio_data,
                    'sampling_rate': sr
                }
            }
            
            # Extract unit
            # Move to device if necessary? Base codecs usually handle 'cpu' default or auto-device
            # But let's force cpu for safety to avoid MPS issues seen earlier
            if hasattr(codec, 'config'):
                # Some codecs might need explicit config call if not in __init__
                pass
            
            if hasattr(codec, 'device'):
                # Force CPU for calculation safety
                 codec.device = 'cpu'
                 if hasattr(codec, 'model'):
                     codec.model.to('cpu')

            with torch.no_grad():
                extracted = codec.extract_unit(data_item)
                unit = extracted.unit
            
            # Calculate TPS
            # unit shape is typically (n_quantizers, T) or (T, n_quantizers) or just (T)
            # We need to find the time dimension.
            # Usually the longest dimension that is not the quantizer count (which is usually small, e.g. 4, 8, 32, 128)
            # WavTokenizer: (1, T) -> T is tokens
            # Encodec: (n_q, T)
            
            shape = unit.shape
            # Heuristic to find Time dimension
            # Usually T is roughly sr / stride
            # codebook dim is usually small < 128
            
            if len(shape) == 1:
                frames = shape[0]
                num_quantizers = 1
            elif len(shape) == 2:
                if shape[0] > shape[1]: # (T, Q)
                    frames = shape[0]
                    num_quantizers = shape[1]
                else: # (Q, T)
                    frames = shape[1]
                    num_quantizers = shape[0]
            elif len(shape) == 3:
                 # (B, Q, T) or (B, T, Q) -> assume B=1 from extract_unit usually returning squeezed
                 # But extract_unit usually returns (Q, T) or (T)
                 # Let's assume (Q, T) mostly
                 frames = max(shape)
                 num_quantizers = shape[0] * shape[1] * shape[2] / frames # Simple check
            else:
                frames = 0
                num_quantizers = 0
            
            tps = frames / duration
            
            # Calculate BPS
            # Depends on codebook size (bits per token)
            # Most codecs use 1024 (10 bits) or 2048 (11 bits) or similar.
            # However, exact bitrate is often defined as:
            # Bitrate = FrameRate * NumQuantizers * BitsPerCode
            # But "BitsPerCode" depends on the model.
            
            # ALTERNATIVE: Use the metric name to guess for some, but user wants calculation.
            # We can't easily know the codebook size from just the unit tensor (it contains indices).
            # But we can assume standard codebook sizes:
            # Encodec: 1024 (10 bits)
            # DAC: 1024 (10 bits)
            # FunCodec: usually 1024?
            
            # Actually, calculating BPS from *tensor size* is tricky without knowing vocab size.
            # But we can print TPS for sure.
            # For BPS, checking the paper/config is safer if we can't inspect the model.
            
            # Let's print TPS first, and try to deduce BPS if possible.
            # For Encodec, we know bits = n_q * 10.
            # BPS (kbps) = TPS * n_q * 10 / 1000
            
            print(f"{metric_name:<40} | {'?':<10} | {tps:<10.2f} (Shape: {shape})")

        except Exception as e:
            print(f"{name:<40} | ERROR      | {e}")

if __name__ == "__main__":
    calculate_metrics()
