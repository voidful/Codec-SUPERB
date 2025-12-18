import argparse
import json
import gc
import os
import time
from datetime import datetime

import numpy as np
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from audiotools import AudioSignal
from SoundCodec.base_codec.general import pad_arrays_to_match
from SoundCodec.metrics import get_metrics
import psutil
from tqdm.contrib.concurrent import process_map


def default_converter(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def compute_metrics(original, model, max_duration):
    original_arrays, resynth_array = pad_arrays_to_match(original['audio']['array'], model['audio']['array'])
    sampling_rate = original['audio']['sampling_rate']
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
        import traceback
        print(f"Error processing entry: {e}")
        # Print debug info for the first few errors to avoid log spam (but since I'm running blind, just print for one and maybe logic to stop spam? No, just print once or catch all)
        # Actually let's print the shape/type of inputs if available
        try:
            print(f"Debug - Original: Type={type(original_iter['audio']['array'])}, Shape={original_iter['audio']['array'].shape}")
            print(f"Debug - Model: Type={type(model_iter['audio']['array'])}, Shape={model_iter['audio']['array'].shape}")
            traceback.print_exc()
        except:
            pass
        return {}


def evaluate_dataset(dataset_name, is_stream, specific_models=None, max_duration=120, max_workers=4, chunksize=10, limit=None):
    start_time = time.time()  # Start time measurement
    print(f"Initial RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB\n")

    if os.path.exists(dataset_name):
        c = load_from_disk(dataset_name)
    else:
        c = load_dataset(dataset_name, streaming=is_stream)
    models = [key for key in c.keys() if key != "original"]

    result_data = {}
    for model in models:
        if specific_models is not None and model not in specific_models:
            continue
        print(f"Evaluating metrics for model: {model}")
        model_start_time = time.time()

        # Process Dataset with Multi-Processing
        args_list = [(original_iter, model_iter, max_duration) for original_iter, model_iter in
                     zip(c['original'], c[model])]
        
        if limit is not None:
            args_list = args_list[:limit]

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

    # Save results
    output_file_name = f"{dataset_name.replace('/', '_')}_evaluation_results.json"
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4, default=default_converter)

    base_filename = f"{args.dataset.replace('/', '_')}_evaluation_results"
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if os.path.exists(f"{base_filename}.json") else ""
    output_file_name = f"{base_filename}{timestamp}.json"

    # Save results to the file
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
