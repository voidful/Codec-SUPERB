import argparse
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets, Features, Dataset
from codec import load_codec, list_codec
from datasets import load_dataset
import dataset as ds_module


def load_dataset_streaming(dataset_name):
    return load_dataset(dataset_name, split='original', streaming=True)


def add_codec_split(cleaned_dataset, codec_name):
    codec = load_codec(codec_name)
    cleaned_dataset = ds_module.general.apply_audio_cast(cleaned_dataset, codec.sampling_rate)
    features = cleaned_dataset.features.copy()
    cleaned_dataset = cleaned_dataset.map(codec.synth, features=features)
    cleaned_dataset = cleaned_dataset.cast_column("audio", Audio(sampling_rate=codec.sampling_rate))
    return cleaned_dataset


def run_experiment(dataset_name, type, add_codec=None, push_to_hub=False, upload_name='AudioDecBenchmark'):
    cleaned_dataset = load_dataset_streaming(dataset_name)
    cleaned_dataset = add_codec_split(cleaned_dataset, add_codec)
    cleaned_dataset = Dataset.from_generator(cleaned_dataset.__iter__)
    print(cleaned_dataset)
    if push_to_hub:
        cleaned_dataset.push_to_hub(
            f"{upload_name}/{dataset_name}_{type}_{add_codec}")  # not able to update existing dataset with streaming, so we create a new one


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process in huggingface/datasets')
    parser.add_argument('--type', required=True, type=str, choices=['synth', 'extract_unit'],
                        help='pick from synth, or extract_unit')
    parser.add_argument('--add_codec', type=str, choices=list_codec(),
                        help='Name of the codec to add to the dataset')
    parser.add_argument('--push_to_hub', required=False, action='store_true')
    parser.add_argument('--upload_name', required=True, default='AudioDecBenchmark')
    args = parser.parse_args()
    run_experiment(args.dataset, args.type, args.add_codec, args.push_to_hub, args.upload_name)
