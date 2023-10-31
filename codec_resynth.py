import argparse
from datasets import DatasetDict, Audio

from codec import load_codec, list_codec
from dataset import load_dataset
import dataset as ds_module


def run_experiment(dataset_name):
    # get sampling rate
    cleaned_dataset = load_dataset(dataset_name)
    d_item = next(iter(cleaned_dataset))
    sampling_rate = d_item['audio']['sampling_rate']
    # reload dataset
    cleaned_dataset = load_dataset(dataset_name)
    datasets_dict = DatasetDict({'original': cleaned_dataset})
    for codec_name in list_codec():
        print(f"Synthesizing dataset with {codec_name}")
        codec = load_codec(codec_name)
        cleaned_dataset_with_audio = ds_module.general.apply_audio_cast(cleaned_dataset, codec.sampling_rate)
        synthesized_dataset = cleaned_dataset_with_audio.map(codec.synth)  # load_from_cache_file=False

        synthesized_dataset = synthesized_dataset.cast_column("audio", Audio(
            sampling_rate=sampling_rate))
        datasets_dict[f'{codec_name}'] = synthesized_dataset

    datasets_dict.push_to_hub(f"AudioDecBenchmark/{dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process in huggingface/datasets')

    args = parser.parse_args()
    run_experiment(args.dataset)
