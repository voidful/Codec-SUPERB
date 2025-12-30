import os
# Fix LLVM SVML symbol errors
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import gc
import time
import soundfile as sf
import io
import numpy as np
import psutil
from datetime import datetime
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from audiotools import AudioSignal
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from SoundCodec.base_codec.general import pad_arrays_to_match
from SoundCodec.metrics import get_metrics
import datasets

# Fix CUDA multiprocessing issues - must be set before any CUDA operations
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set
import json
import gc
import os
import time
import soundfile as sf
import io
import numpy as np
import psutil
from datetime import datetime
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from audiotools import AudioSignal
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from SoundCodec.base_codec.general import pad_arrays_to_match
from SoundCodec.metrics import get_metrics
import datasets

# The previous hacks are removed as they were ineffective against datasets behavior.


def default_converter(o):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def safe_load_audio(audio_entry):
    """
    Safely load audio array and sampling rate from a non-decoded datasets Audio feature.
    Returns: (array [T, C], sampling_rate) or (None, None) if failed.
    """
    try:
        # If it's already a dict with array (decoded)
        if isinstance(audio_entry, dict) and 'array' in audio_entry:
            array = audio_entry['array']
            sr = audio_entry['sampling_rate']
            
            # Critical: Ensure array is numpy on CPU for multiprocessing
            # CUDA tensors cannot be serialized across processes
            if hasattr(array, 'cpu'):
                array = array.cpu().numpy()
            elif not isinstance(array, np.ndarray):
                array = np.array(array)
            
            return array, sr

        # If it's the raw dict (decode=False)
        if isinstance(audio_entry, dict):
            audio_bytes = audio_entry.get('bytes')
            path = audio_entry.get('path')
            if audio_bytes:
                array, sr = sf.read(io.BytesIO(audio_bytes))
                return array, sr
            elif path:
                array, sr = sf.read(path)
                return array, sr

        return None, None
    except Exception as e:
        print(f"Error in safe_load_audio: {e}")
        return None, None


def compute_metrics(original, model, max_duration, save_audio=False):
    orig_array, orig_sr = safe_load_audio(original['audio'])
    model_array, model_sr = safe_load_audio(model['audio'])
    
    if orig_array is None or model_array is None:
        return None 
    
    # Check sampling rate mismatch
    if orig_sr != model_sr:
        print(f"Warning: Sampling rate mismatch - original: {orig_sr}Hz, model: {model_sr}Hz")
        return None

    original_arrays, resynth_array = pad_arrays_to_match(orig_array, model_array)
    sampling_rate = orig_sr
    original_signal = AudioSignal(original_arrays, sampling_rate)
    if original_signal.duration > max_duration:
        return None
    model_signal = AudioSignal(resynth_array, sampling_rate)
    metrics = get_metrics(original_signal, model_signal)
    
    # Optionally include audio data
    if save_audio:
        return {
            'metrics': metrics,
            'original_audio': original_arrays.tolist() if isinstance(original_arrays, np.ndarray) else original_arrays,
            'reconstructed_audio': resynth_array.tolist() if isinstance(resynth_array, np.ndarray) else resynth_array,
            'sampling_rate': sampling_rate
        }
    
    return metrics


def process_entry(args):
    original_iter, model_iter, max_duration, save_audio = args
    try:
        result = compute_metrics(original_iter, model_iter, max_duration, save_audio)
        if result is not None:
            return result
        else:
            return {}
    except Exception as e:
        print(f"Error processing entry: {e}")
        return {}


def evaluate_dataset(dataset_name, is_stream, specific_models=None, max_duration=120, max_workers=4, chunksize=10, limit=None, save_audio=True):
    start_time = time.time()
    print(f"Initial RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB\n")

    if os.path.exists(dataset_name):
        c = load_from_disk(dataset_name)
    else:
        c = load_dataset(dataset_name, streaming=is_stream)
    
    # Check if this is a pre-synthesized dataset (has 'original' split) or original dataset
    is_presynthesized = 'original' in c.keys()
    
    if is_presynthesized:
        # Original mode: dataset already has original and codec splits
        print("Using pre-synthesized dataset mode")
        # Disable automatic decoding to bypass torchcodec
        for split in c.keys():
            c[split] = c[split].cast_column("audio", datasets.Audio(decode=False))
        
        models = [key for key in c.keys() if key != "original"]
        
        # Cache original entries
        print("Caching original dataset entries...")
        original_entries = list(c['original'])
        if limit is not None:
            original_entries = original_entries[:limit]
        print(f"Cached {len(original_entries)} original entries")
    else:
        # New mode: apply codecs on-the-fly to original dataset
        print("Using direct codec evaluation mode")
        if specific_models is None or len(specific_models) == 0:
            raise ValueError("Must specify --models when using direct evaluation mode")
        
        models = specific_models
        
        # Combine all splits into one dataset
        all_entries = []
        for split_name in c.keys():
            try:
                split_data = c[split_name]
                # For direct evaluation, we need decoded audio for codec.synth()
                # Use decode=True but with error handling for LLVM issues
                split_data = split_data.cast_column("audio", datasets.Audio(decode=True))
                split_list = list(split_data)
                # Add category field if not present
                for entry in split_list:
                    if 'category' not in entry:
                        entry['category'] = split_name
                all_entries.extend(split_list)
            except Exception as e:
                error_msg = str(e)
                if 'LLVM' in error_msg or 'svml' in error_msg.lower():
                    print(f"\n{'='*80}")
                    print("LLVM/MKL Error Detected!")
                    print(f"{'='*80}")
                    print("\nThis is a library compatibility issue. Try these solutions:")
                    print("\n1. Use pre-synthesized dataset instead:")
                    print("   python3 scripts/dataset_creator.py --dataset voidful/codec-superb-tiny --specific_codecs auv llmcodec bigcodec_1k")
                    print("   python3 scripts/benchmarking.py --dataset datasets/voidful/codec-superb-tiny_synth --models auv llmcodec bigcodec_1k")
                    print("\n2. Or reinstall PyTorch with different BLAS:")
                    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
                    print(f"\n{'='*80}\n")
                raise
        
        if limit is not None:
            all_entries = all_entries[:limit]
        
        original_entries = all_entries
        print(f"Loaded {len(original_entries)} samples from {len(c.keys())} splits")
    
    # Warn about memory usage for large datasets
    if len(original_entries) > 5000:
        print(f"Warning: Large dataset ({len(original_entries)} samples) - high memory usage expected")

    result_data = {}
    for model in models:
        if specific_models is not None and model not in specific_models:
            continue
        
        # Check if this is an encode-only codec by trying to load it
        try:
            from SoundCodec.codec import load_codec
            codec_instance = load_codec(model)
            if hasattr(codec_instance, 'supports_decode') and not codec_instance.supports_decode:
                print(f"Skipping {model}: encode-only codec (no decoder available)")
                result_data[model] = {
                    "encode_only": True,
                    "message": "This codec only supports encoding. No reconstruction metrics available."
                }
                del codec_instance
                gc.collect()
                continue
            del codec_instance
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not check codec {model}: {e}")
        
        print(f"Evaluating metrics for model: {model}")
        model_start_time = time.time()

        if is_presynthesized:
            # Original mode: use pre-synthesized model entries
            model_entries = list(c[model])
            if limit is not None:
                model_entries = model_entries[:limit]

            # Process Dataset with Multi-Processing
            args_list = [(original_iter, model_iter, max_duration, save_audio) for original_iter, model_iter in
                         zip(original_entries, model_entries)]
        else:
            # New mode: apply codec on-the-fly
            from SoundCodec.codec import load_codec
            print(f"Loading codec: {model}")
            codec_instance = load_codec(model)
            
            # Create model entries by encoding and decoding original audio
            print(f"Encoding and decoding {len(original_entries)} samples with {model}...")
            model_entries = []
            for entry in tqdm(original_entries, desc=f"Synthesizing with {model}"):
                try:
                    # Get original sampling rate before synthesis
                    original_sr = None
                    if 'audio' in entry:
                        if isinstance(entry['audio'], dict):
                            original_sr = entry['audio'].get('sampling_rate')
                        
                    # Synthesize audio using the codec
                    synthesized = codec_instance.synth(entry, local_save=False)
                    
                    # Ensure sampling_rate is preserved in synthesized output
                    if 'audio' in synthesized:
                        if not isinstance(synthesized['audio'], dict):
                            synthesized['audio'] = {'array': synthesized['audio'], 'sampling_rate': original_sr}
                        elif synthesized['audio'].get('sampling_rate') is None and original_sr is not None:
                            synthesized['audio']['sampling_rate'] = original_sr
                    
                    model_entries.append(synthesized)
                except Exception as e:
                    print(f"Error synthesizing sample: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add empty entry to maintain alignment
                    model_entries.append({'audio': {'array': None, 'sampling_rate': None}})
            
            # Process Dataset with Multi-Processing
            args_list = [(original_iter, model_iter, max_duration, save_audio) for original_iter, model_iter in
                         zip(original_entries, model_entries)]
            
            # Clean up codec instance
            del codec_instance
            gc.collect()

        # Use sequential processing if multiprocessing is disabled or if max_workers is 1
        if max_workers == 1:
            print(f"Using sequential processing (max_workers=1)")
            metrics_results = []
            for args in tqdm(args_list, desc=f"Processing {model}"):
                result = process_entry(args)
                metrics_results.append(result)
        else:
            # Use multiprocessing
            metrics_results = process_map(process_entry, args_list, max_workers=max_workers, chunksize=chunksize)
        metrics_results = [metrics for metrics in metrics_results if metrics is not None]
        # Process Dataset END

        # Aggregate the metrics
        aggregated_metrics = defaultdict(lambda: defaultdict(list))
        audio_samples = [] if save_audio else None
        failed_count = 0
        
        for idx, (result, entry) in enumerate(zip(metrics_results, original_entries)):
            if result:
                category = entry.get('category', 'overall')
                
                # Handle both formats: plain metrics dict or dict with 'metrics' key
                if save_audio and 'metrics' in result:
                    metrics = result['metrics']
                    # Store audio sample with metadata
                    audio_samples.append({
                        'id': entry.get('id', f'sample_{idx}'),
                        'category': category,
                        'original_audio': result['original_audio'],
                        'reconstructed_audio': result['reconstructed_audio'],
                        'sampling_rate': result['sampling_rate']
                    })
                else:
                    metrics = result
                
                for k, v in metrics.items():
                    aggregated_metrics[category][k].append(v)
                    aggregated_metrics['overall'][k].append(v)
            else:
                failed_count += 1

        # Calculate average metrics per category
        model_result = {}
        for category, metrics_dict in aggregated_metrics.items():
            model_result[category] = {k: np.nanmean(v) if v else np.nan for k, v in metrics_dict.items()}
        
        # Add audio samples if save_audio is enabled
        if save_audio and audio_samples:
            model_result['audio_samples'] = audio_samples
        
        result_data[model] = model_result
        
        # Report statistics
        total_samples = len(metrics_results)
        success_count = total_samples - failed_count
        print(f"Processed: {success_count}/{total_samples} samples successfully ({failed_count} failed)")
        gc.collect()
        print(f"RAM used after processing {model}: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        print(f"Time taken for {model}: {time.time() - model_start_time:.2f} seconds")
        
        # Print metrics summary (exclude audio_samples to avoid verbose output)
        print("\nMetrics Summary:")
        for category, metrics in model_result.items():
            if category != 'audio_samples':
                print(f"  {category}: {metrics}")
        if save_audio and 'audio_samples' in model_result:
            print(f"  Audio samples saved: {len(model_result['audio_samples'])} samples")
        print()

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Final RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Save results with timestamp if file already exists
    # Sanitize dataset name for filename (remove invalid characters)
    safe_dataset_name = dataset_name.replace('/', '_').replace(':', '_').replace('?', '_').replace('*', '_')
    base_filename = f"{safe_dataset_name}_evaluation_results"
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if os.path.exists(f"{base_filename}.json") else ""
    output_file_name = f"{base_filename}{timestamp}.json"

    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4, default=default_converter)

    print(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate audio datasets.')
    parser.add_argument('--dataset', type=str, default="AudioDecBenchmark/librispeech_asr_dummy_synth",
                        help='Name of the dataset to evaluate')
    parser.add_argument('--streaming', action='store_true', help='Evaluate in streaming mode')
    parser.add_argument('--models', nargs='*', help='Specific models to evaluate')
    parser.add_argument('--max_duration', type=int, default=120,
                        help='Maximum duration of audio recordings in seconds')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of workers for multi-processing')
    parser.add_argument('--chunksize', type=int, default=10, help='Chunk size for multi-processing')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')
    parser.add_argument('--no_save_audio', action='store_false', dest='save_audio',
                        help='Disable saving original and reconstructed audio samples (saves by default)')

    args = parser.parse_args()
    
    # Handle comma-separated models (e.g., --models auv,llmcodec,bigcodec_1k)
    if args.models and len(args.models) == 1 and ',' in args.models[0]:
        args.models = [m.strip() for m in args.models[0].split(',')]
    
    evaluate_dataset(args.dataset, args.streaming, args.models, args.max_duration, args.max_workers, args.chunksize, args.limit, args.save_audio)
