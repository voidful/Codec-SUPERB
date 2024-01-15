import argparse
import json
import gc  # Garbage Collection

import numpy as np
from datasets import load_dataset, DownloadMode
from collections import defaultdict
from audiotools import AudioSignal
from codec.general import pad_arrays_to_match
from metrics import get_metrics
import psutil


def compute_metrics(entry, id_dict):
    original_arrays, resynth_array = pad_arrays_to_match(entry['audio']['array'], id_dict[entry['id']])
    sampling_rate = entry['audio']['sampling_rate']
    original_signal = AudioSignal(original_arrays, sampling_rate)
    model_signal = AudioSignal(resynth_array, sampling_rate)
    metrics = get_metrics(original_signal, model_signal)
    return metrics


def batched_dataset(dataset, batch_size):
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_entry(entry, id_dict, metrics_results):
    if isinstance(entry, dict):  # Single entry in streaming mode
        metrics = compute_metrics(entry, id_dict)
        metrics_results.append(metrics)
    elif isinstance(entry, list):  # Batch of entries in batch mode
        for item in entry:
            metrics = compute_metrics(item, id_dict)
            metrics_results.append(metrics)


def evaluate_dataset(dataset_name, mode, batch_size):
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    c = load_dataset(dataset_name, streaming=(mode == 'streaming'))
    models = [key for key in c.keys() if key != "original"]
    result_data = {}

    for model in models:
        print(f"Evaluating metrics for model: {model}")
        id_dict = {i['id']: i['audio']['array'] for i in c[model]}

        # Process dataset
        metrics_results = []
        dataset_iterable = c['original'] if mode == 'streaming' else batched_dataset(c['original'], batch_size)

        for entry in dataset_iterable:
            process_entry(entry, id_dict, metrics_results)

        # Aggregate the metrics
        aggregated_metrics = defaultdict(list)
        for metrics in metrics_results:
            for k, v in metrics.items():
                aggregated_metrics[k].append(v)

        # Calculate and print average metrics
        model_result = {k: np.nanmean(v) if v else np.nan for k, v in aggregated_metrics.items()}
        result_data[model] = model_result
        del id_dict  # Release memory
        gc.collect()  # Explicitly invoke garbage collection
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Save results
    output_file_name = f"{dataset_name.replace('/', '_')}_evaluation_results.json"
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4)

    print(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate audio datasets.')
    parser.add_argument('--dataset', type=str, default="AudioDecBenchmark/librispeech_asr_dummy_synth",
                        help='Name of the dataset to evaluate')
    parser.add_argument('--mode', type=str, choices=['batch', 'streaming'], default='streaming',
                        help='Mode of dataset loading: batch or streaming')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing the dataset')

    args = parser.parse_args()
    evaluate_dataset(args.dataset, args.mode, args.batch_size)
