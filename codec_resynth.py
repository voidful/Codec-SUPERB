import argparse
from datasets import DatasetDict, Audio

from codec import load_codec
from dataset import load_dataset
import dataset as ds_module


def run_experiment(codec_name, dataset_name):
    codec = load_codec(codec_name)
    cleaned_dataset = load_dataset(dataset_name)

    cleaned_dataset_with_audio = ds_module.general.apply_audio_cast(cleaned_dataset, codec.sampling_rate)
    synthesized_dataset = cleaned_dataset_with_audio.map(codec.synth)

    # Construct DatasetDict and push to hub
    datasets_dict = DatasetDict({
        'original': cleaned_dataset,
        f'{codec_name}': synthesized_dataset.cast_column("audio",
                                                         Audio(sampling_rate=cleaned_dataset['audio'][0][
                                                             'sampling_rate']))
    })

    datasets_dict.push_to_hub(f"AudioDecBenchmark/{dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--codec', type=str, required=True,
                        help='Name of the codec to use (e.g., "facebook/encodec_24khz")')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process (e.g., "hf-internal-testing/librispeech_asr_dummy")')

    args = parser.parse_args()

    run_experiment(args.codec, args.dataset)
