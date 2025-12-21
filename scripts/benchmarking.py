import argparse
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
    if isinstance(o, np.float32):
        return float(o)
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


def compute_metrics(original, model, max_duration):
    orig_array, orig_sr = safe_load_audio(original['audio'])
    model_array, model_sr = safe_load_audio(model['audio'])
    
    if orig_array is None or model_array is None:
        return None 

    original_arrays, resynth_array = pad_arrays_to_match(orig_array, model_array)
    sampling_rate = orig_sr
    original_signal = AudioSignal(original_arrays, sampling_rate)
    if original_signal.duration > max_duration:
        return None
    model_signal = AudioSignal(resynth_array, sampling_rate)
    metrics = get_metrics(original_signal, model_signal)
    return metrics


def process_entry(args):
    original_iter, model_iter, max_duration = args
    try:
        metrics = compute_metrics(original_iter, model_iter, max_duration)
        if metrics is not None:
            return metrics
        else:
            return {}
    except Exception as e:
        print(f"Error processing entry: {e}")
        return {}


def evaluate_dataset(dataset_name, is_stream, specific_models=None, max_duration=120, max_workers=4, chunksize=10, limit=None):
    start_time = time.time()
    print(f"Initial RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB\n")

    if os.path.exists(dataset_name):
        c = load_from_disk(dataset_name)
    else:
        c = load_dataset(dataset_name, streaming=is_stream)
    
    # Disable automatic decoding to bypass torchcodec
    for split in c.keys():
        c[split] = c[split].cast_column("audio", datasets.Audio(decode=False))

    models = [key for key in c.keys() if key != "original"]

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

        # Process Dataset with Multi-Processing
        args_list = [(original_iter, model_iter, max_duration) for original_iter, model_iter in
                     zip(c['original'], c[model])]
        
        if limit is not None:
            args_list = args_list[:limit]

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
        for metrics, entry in zip(metrics_results, c['original']):
            if metrics:
                category = entry.get('category', 'overall')
                for k, v in metrics.items():
                    aggregated_metrics[category][k].append(v)
                    aggregated_metrics['overall'][k].append(v)

        # Calculate average metrics per category
        model_result = {}
        for category, metrics_dict in aggregated_metrics.items():
            model_result[category] = {k: np.nanmean(v) if v else np.nan for k, v in metrics_dict.items()}
        
        result_data[model] = model_result
        gc.collect()
        print(f"RAM used after processing {model}: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        print(f"Time taken for {model}: {time.time() - model_start_time:.2f} seconds")
        print(model_result)
        print()

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Final RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Save results with timestamp if file already exists
    base_filename = f"{dataset_name.replace('/', '_')}_evaluation_results"
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
    parser.add_argument('--batch', type=int, default=100,
                        help='Batch size for processing the dataset')
    parser.add_argument('--models', nargs='*', help='Specific models to evaluate')
    parser.add_argument('--max_duration', type=int, default=120,
                        help='Maximum duration of audio recordings in seconds')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of workers for multi-processing')
    parser.add_argument('--chunksize', type=int, default=10, help='Chunk size for multi-processing')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')

    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.streaming, args.models, args.max_duration, args.max_workers, args.chunksize, args.limit)
