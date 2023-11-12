import argparse
import json

from datasets import load_dataset, DownloadMode
from collections import defaultdict
from audiotools import AudioSignal

from codec import list_codec
from codec.general import pad_arrays_to_match
from metrics import get_metrics


def evaluate_dataset(dataset_name):
    c = load_dataset(dataset_name, download_mode=DownloadMode.FORCE_REDOWNLOAD)

    models = [key for key in c.keys() if key != "original"]

    result_data = {}
    for model in models:
        if model != "original":
            print(f"Evaluating metrics for model: {model}")

            id_dict = {i['id']: i['audio']['array'] for i in c[model]}

            def compute_metrics(entry):
                original_arrays, resynth_array = pad_arrays_to_match(entry['audio']['array'], id_dict[entry['id']])
                sampling_rate = entry['audio']['sampling_rate']
                original_signal = AudioSignal(original_arrays, sampling_rate)
                model_signal = AudioSignal(resynth_array, sampling_rate)
                metrics = get_metrics(original_signal, model_signal)
                entry['metrics'] = metrics
                return entry

            cal_metrics_ds = c['original'].map(compute_metrics, num_proc=1, load_from_cache_file=False)
            # cal_metrics_ds = compute_metrics(c['original'])

            result_dict = defaultdict(list)
            for ds_item in cal_metrics_ds:
                for k, v in ds_item['metrics'].items():
                    result_dict[k].append(v)

            model_result = {}
            for k, v in result_dict.items():
                avg_metric = sum(v) / len(v)
                print(f"{k}: {avg_metric}")
                model_result[k] = avg_metric
            result_data[model] = model_result

    output_file_name = f"{dataset_name.replace('/', '_')}_evaluation_results.json"
    with open(output_file_name, 'w') as out_file:
        json.dump(result_data, out_file, indent=4)
    print(f"Results saved to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate audio datasets.')
    parser.add_argument('--dataset', type=str, default="AudioDecBenchmark/librispeech_asr_dummy",
                        help='Name of the dataset to evaluate')

    args = parser.parse_args()

    evaluate_dataset(args.dataset)
