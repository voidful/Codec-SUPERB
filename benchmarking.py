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
from base_codec.general import pad_arrays_to_match
from metrics import get_metrics
import psutil
from tqdm.auto import tqdm


def compute_metrics(original, model, max_duration):
    original_arrays, resynth_array = pad_arrays_to_match(original['audio']['array'], model['audio']['array'])
    sampling_rate = original['audio']['sampling_rate']
    original_signal = AudioSignal(original_arrays, sampling_rate)
    if original_signal.duration > max_duration:
        return None
    model_signal = AudioSignal(resynth_array, sampling_rate)
    metrics = get_metrics(original_signal, model_signal)
    return metrics


def process_entry(original_iter, model_iter, metrics_results, max_duration):
    metrics = compute_metrics(original_iter, model_iter, max_duration)
    if metrics is not None:
        metrics_results.append(metrics)


def evaluate_dataset(dataset_name, is_stream, specific_models=None, max_duration=120):
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

        # Process dataset
        metrics_results = []
        for original_iter, model_iter in tqdm(zip(c['original'], c[model])):
            process_entry(original_iter, model_iter, metrics_results, max_duration)

        # Aggregate the metrics
        aggregated_metrics = defaultdict(list)
        for metrics in metrics_results:
            for k, v in metrics.items():
                aggregated_metrics[k].append(v)

        # Calculate and print average metrics
        model_result = {k: np.nanmean(v) if v else np.nan for k, v in aggregated_metrics.items()}
        result_data[model] = model_result
        gc.collect()  # Explicitly invoke garbage collection
        print(f"RAM used after processing {model}: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        print(f"Time taken for {model}: {time.time() - model_start_time:.2f} seconds")
        print(model_result)
        print()

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Final RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Save results
    output_file_name = f"{dataset_name.replace('/', '_')}_evaluation_results.json"
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4)

    base_filename = f"{args.dataset.replace('/', '_')}_evaluation_results"
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if os.path.exists(f"{base_filename}.json") else ""
    output_file_name = f"{base_filename}{timestamp}.json"

    # Save results to the file
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4)

    print(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate audio datasets.')
    parser.add_argument('--dataset', type=str, default="AudioDecBenchmark/librispeech_asr_dummy_synth",
                        help='Name of the dataset to evaluate')
    parser.add_argument('--streaming', action='store_true', default=True, help='Evaluate in streaming mode')
    parser.add_argument('--batch', type=int, default=100,
                        help='Batch size for processing the dataset')
    parser.add_argument('--models', nargs='*', help='Specific models to evaluate')
    parser.add_argument('--max_duration', type=int, default=120,
                        help='Maximum duration of audio recordings in seconds')

    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.streaming, args.models, args.max_duration)
