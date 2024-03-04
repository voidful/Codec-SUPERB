import argparse
from datasets import Audio, Dataset, Value, Sequence
from SoundCodec.codec import load_codec, list_codec
from SoundCodec.dataset.general import apply_audio_cast
from datasets import load_dataset


def load_dataset_streaming(dataset_name):
    return load_dataset(dataset_name, split='original', streaming=True)


def add_codec_split(cleaned_dataset, codec_name):
    codec = load_codec(codec_name)
    original_sampling_rate = cleaned_dataset.features['audio'].sampling_rate
    cleaned_dataset = apply_audio_cast(cleaned_dataset, codec.sampling_rate)
    features = cleaned_dataset.features.copy()
    features['unit'] = Sequence(feature=Sequence(feature=Value(dtype='int64', id=None)))
    cleaned_dataset = cleaned_dataset.map(codec.synth, features=features)
    return cleaned_dataset, original_sampling_rate


def run_experiment(dataset_name, update_codec, extract_unit_only, push_to_hub):
    cleaned_dataset = load_dataset_streaming(dataset_name)
    cleaned_dataset, original_sampling_rate = add_codec_split(cleaned_dataset, update_codec)
    cleaned_dataset = Dataset.from_generator(cleaned_dataset.__iter__)
    if push_to_hub:
        if "_synth" in dataset_name:
            dataset_name = dataset_name.replace("_synth", "")
        datasets_dict_unit_only = cleaned_dataset.remove_columns(['audio'])
        datasets_dict_unit_only.push_to_hub(
            f"{dataset_name}_unit", split=update_codec)
        if not extract_unit_only:
            cleaned_dataset = cleaned_dataset.remove_columns(['unit'])
            cleaned_dataset = cleaned_dataset.cast_column("audio", Audio(sampling_rate=original_sampling_rate))
            cleaned_dataset.push_to_hub(
                f"{dataset_name}_synth", split=update_codec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process in huggingface/datasets')
    parser.add_argument('--update_codec', type=str, choices=list_codec(),
                        help='Name of the codec to add to the dataset')
    parser.add_argument('--extract_unit_only', required=False, action='store_true')
    parser.add_argument('--push_to_hub', required=False, action='store_true')
    args = parser.parse_args()
    run_experiment(args.dataset, args.update_codec, args.extract_unit_only, args.push_to_hub)
