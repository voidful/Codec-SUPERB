import argparse
from datasets import DatasetDict, Audio, load_from_disk
from codec import load_codec, list_codec
from dataset import load_dataset
import dataset as ds_module
from dataset.general import extract_unit


def run_experiment(dataset_name):
    cleaned_dataset = load_dataset(dataset_name)
    d_item = next(iter(cleaned_dataset))
    sampling_rate = d_item['audio']['sampling_rate']
    cleaned_dataset = load_dataset(dataset_name)
    if args.type == 'synth':
        datasets_dict = DatasetDict({'original': cleaned_dataset})
    else:
        datasets_dict = DatasetDict({})
    for codec_name in list_codec():
        print(f"Synthesizing dataset with {codec_name}")
        # load from disk if already synthesized
        try:
            synthesized_dataset = load_from_disk(f"./datasets/{dataset_name}_{codec_name}_{args.type}/")
            datasets_dict[f'{codec_name}'] = synthesized_dataset
            continue
        except:
            pass
        codec = load_codec(codec_name)
        synthesized_dataset = ds_module.general.apply_audio_cast(cleaned_dataset, codec.sampling_rate)
        if args.type == 'extract_unit':
            synthesized_dataset = synthesized_dataset.map(extract_unit, fn_kwargs={'extract_unit_class': codec})
        else:
            synthesized_dataset = synthesized_dataset.map(codec.synth)
            synthesized_dataset = synthesized_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        synthesized_dataset.save_to_disk(f"./datasets/{dataset_name}_{codec_name}_{args.type}/")
        datasets_dict[f'{codec_name}'] = synthesized_dataset

    if args.type == 'extract_unit':
        datasets_dict = datasets_dict.remove_columns(['audio'])
    datasets_dict.save_to_disk(f"./datasets/{dataset_name}_{args.type}")
    if args.push_to_hub:
        push_to_hub_org = args.upload_name
        datasets_dict.push_to_hub(f"{push_to_hub_org}/{dataset_name}_{args.type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process in huggingface/datasets')
    parser.add_argument('--type', required=True, type=str, choices=['synth', 'extract_unit'],
                        help='pick from synth, or extract_unit')
    parser.add_argument('--push_to_hub', required=False, action='store_true')
    parser.add_argument('--upload_name', required=False, default='Codec-SUPERB')
    args = parser.parse_args()
    run_experiment(args.dataset)
