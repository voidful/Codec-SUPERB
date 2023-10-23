import argparse
from datasets import load_dataset
from collections import defaultdict
from audiotools import AudioSignal

from metrics import get_metrics


def evaluate_dataset(dataset_name, output_file):
    c = load_dataset(dataset_name)

    models = [key for key in c.keys() if key != "original"]

    with open(output_file, 'w') as out_file:
        for model in models:
            print(f"Evaluating metrics for model: {model}")
            out_file.write(f"Evaluating metrics for model: {model}\n")

            id_dict = {i['id']: i['audio']['array'] for i in c[model]}
            result_dict = defaultdict(list)

            def compute_metrics(entry):
                original_signal = AudioSignal(entry['audio']['array'], entry['audio']['sampling_rate'])
                model_signal = AudioSignal(id_dict[entry['id']], entry['audio']['sampling_rate'])
                metrics = get_metrics(original_signal, model_signal)
                for k, v in metrics.items():
                    result_dict[k].append(v)
                return entry

            c['original'].map(compute_metrics, num_proc=10)

            for k, v in result_dict.items():
                avg_metric = sum(v) / len(v)
                print(f"{k}: {avg_metric}")
                out_file.write(f"{k}: {avg_metric}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate audio datasets.')
    parser.add_argument('--dataset', type=str, default="AudioDecBenchmark/librispeech_asr_dummy",
                        help='Name of the dataset to evaluate')
    parser.add_argument('--output', type=str, default="results.txt", help='File to output the evaluation results')

    args = parser.parse_args()

    evaluate_dataset(args.dataset, args.output)
