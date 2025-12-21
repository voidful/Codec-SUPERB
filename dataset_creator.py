import argparse
from datasets import DatasetDict, Audio, load_from_disk
import datasets
from SoundCodec.codec import load_codec, list_codec
from SoundCodec.dataset import load_dataset
from SoundCodec.dataset.general import apply_audio_cast, extract_unit
import numpy as np
import io
import soundfile as sf


def manual_decode_item(x):
    try:
        audio_entry = x['audio']
        if isinstance(audio_entry, dict):
            audio_bytes = audio_entry.get('bytes')
            path = audio_entry.get('path')
            if audio_bytes:
                with io.BytesIO(audio_bytes) as f:
                    array, sr = sf.read(f)
            elif path:
                array, sr = sf.read(path)
            else:
                array, sr = None, None
            
            x['audio'] = {
                'array': array,
                'sampling_rate': sr
            }
        return x
    except Exception as e:
        print(f"Error in manual_decode_item: {e}")
        return x


def get_audio_info(x):
    try:
        audio_entry = x['audio']
        if isinstance(audio_entry, dict):
            audio_bytes = audio_entry.get('bytes')
            path = audio_entry.get('path')
            if audio_bytes:
                with io.BytesIO(audio_bytes) as f:
                    info = sf.info(f)
                    return info.duration, info.samplerate
            elif path:
                info = sf.info(path)
                return info.duration, info.samplerate
        return 0, None
    except Exception:
        return 0, None


def run_experiment(dataset_name):
    cleaned_dataset = load_dataset(dataset_name)
    if args.limit:
        cleaned_dataset = cleaned_dataset.select(range(min(args.limit, len(cleaned_dataset))))
    
    # Disable automatic decoding to avoid C++ backend crashes during mass filtering and synthesis
    cleaned_dataset = cleaned_dataset.cast_column("audio", Audio(decode=False))
    
    print("before filter duration", cleaned_dataset)
    cleaned_dataset = cleaned_dataset.filter(
        lambda x: get_audio_info(x)[0] <= args.max_duration)
    
    print("after filter duration", cleaned_dataset)
    
    # Get sampling rate safely from the first item
    first_item = cleaned_dataset[0]
    _, sampling_rate = get_audio_info(first_item)
    
    # Determine which codecs to process
    if args.specific_codecs:
        # User specified specific codecs to add
        codecs_to_process = args.specific_codecs
        print(f"Processing specific codecs: {codecs_to_process}")
        
        # Try to load existing dataset
        try:
            if args.extract_unit_only:
                existing_dataset_path = f"./datasets/{dataset_name}_unit"
            else:
                existing_dataset_path = f"./datasets/{dataset_name}_synth"
            
            print(f"Loading existing dataset from {existing_dataset_path}")
            datasets_dict = load_from_disk(existing_dataset_path)
            print(f"Loaded existing dataset with splits: {list(datasets_dict.keys())}")
            
            # If not extract_unit_only and 'original' is not in the dataset, add it
            if not args.extract_unit_only and 'original' not in datasets_dict:
                datasets_dict['original'] = cleaned_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
                print("Added 'original' split to dataset")
        except Exception as e:
            print(f"Could not load existing dataset ({e}), creating new one")
            if not args.extract_unit_only:
                datasets_dict = DatasetDict({'original': cleaned_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))})
            else:
                datasets_dict = DatasetDict({})
    else:
        # Process all codecs (original behavior)
        codecs_to_process = list_codec()
        print(f"Processing all {len(codecs_to_process)} codecs")
        
        if not args.extract_unit_only:
            datasets_dict = DatasetDict({'original': cleaned_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))})
        else:
            datasets_dict = DatasetDict({})
    
    for codec_name in codecs_to_process:
        print(f"Synthesizing dataset with {codec_name}")
        
        # Skip if already exists in the dataset (unless force regenerate)
        if codec_name in datasets_dict and not args.force_regenerate:
            print(f"  Skipping {codec_name} - already exists in dataset (use --force_regenerate to override)")
            continue
        
        # load from disk if already synthesized
        try:
            synthesized_dataset = load_from_disk(f"./cached_datasets/{dataset_name}_{codec_name}/")
            datasets_dict[f'{codec_name}'] = synthesized_dataset
            print(f"  Loaded {codec_name} from cache")
            continue
        except:
            pass
        
        try:
            codec = load_codec(codec_name)
        except Exception as e:
            if args.only_1d and ("unicodec" in codec_name or "auv" in codec_name):
                print(f"Skipping {codec_name} for 1D check as requested/failed load: {e}")
                continue
            print(f"Error loading codec {codec_name}: {e}")
            continue

        if args.only_1d:
            try:
                if "sqcodec" in codec_name:
                    print(f"Skipping {codec_name} because it is treated as not 1D (exception)")
                    continue
                if not codec.is_1d() and "auv" not in codec_name:
                    print(f"Skipping {codec_name} because it is not 1D")
                    continue
                elif not codec.is_1d() and "auv" in codec_name:
                    print(f"Forcing {codec_name} to be treated as 1D")
            except Exception as e:
                print(f"Skipping {codec_name} due to extraction error during 1D check: {e}")
                continue

        if args.extract_unit_only:
            synthesized_dataset = cleaned_dataset.map(lambda x: extract_unit(manual_decode_item(x), codec))
        else:
            synthesized_dataset = cleaned_dataset.map(lambda x: codec.synth(manual_decode_item(x)))
            synthesized_dataset = synthesized_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        synthesized_dataset.save_to_disk(f"./cached_datasets/{dataset_name}_{codec_name}/")
        datasets_dict[f'{codec_name}'] = synthesized_dataset
        print(f"  Added {codec_name} to dataset")

    # Save unit-only dataset
    datasets_dict_unit_only = datasets_dict.remove_columns(['audio'])
    if 'original' in datasets_dict_unit_only:
        datasets_dict_unit_only.pop('original')
    datasets_dict_unit_only.save_to_disk(f"./datasets/{dataset_name}_unit")
    print(f"Saved unit-only dataset to ./datasets/{dataset_name}_unit")
    
    # remove datasets_dict columns if they have 'unit', and use datasets_dict_synth for saving
    datasets_dict_synth = DatasetDict({})
    for key in datasets_dict.keys():
        if 'unit' not in datasets_dict[key].column_names:
            datasets_dict_synth[key] = datasets_dict[key]
        else:
            datasets_dict_synth[key] = datasets_dict[key].remove_columns(['unit'])
    if not args.extract_unit_only:
        datasets_dict_synth.save_to_disk(f"./datasets/{dataset_name}_synth")
        print(f"Saved synth dataset to ./datasets/{dataset_name}_synth")

    if args.push_to_hub:
        push_to_hub_org = args.upload_name
        if not args.extract_unit_only:
            datasets_dict_synth.push_to_hub(f"{push_to_hub_org}/{dataset_name}_synth")
            print(f"Pushed synth dataset to {push_to_hub_org}/{dataset_name}_synth")
        datasets_dict_unit_only.push_to_hub(f"{push_to_hub_org}/{dataset_name}_unit")
        print(f"Pushed unit dataset to {push_to_hub_org}/{dataset_name}_unit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset to process in huggingface/datasets')
    parser.add_argument('--extract_unit_only', required=False, action='store_true',
                        help='Only extract units without synthesis')
    parser.add_argument('--max_duration', required=False, type=int, default=120,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--push_to_hub', required=False, action='store_true',
                        help='Push the dataset to HuggingFace Hub')
    parser.add_argument('--upload_name', required=False, default='Codec-SUPERB',
                        help='Organization name for HuggingFace Hub upload')
    parser.add_argument('--limit', required=False, type=int, default=None,
                        help='Limit number of samples to process')
    parser.add_argument('--only_1d', required=False, action='store_true',
                        help='Only process 1D codecs')
    parser.add_argument('--specific_codecs', required=False, nargs='+', default=None,
                        help='Process only specific codec(s), e.g., --specific_codecs s3tokenizer_v1 s3tokenizer_v2_25hz')
    parser.add_argument('--force_regenerate', required=False, action='store_true',
                        help='Force regenerate codecs even if they already exist in the dataset')
    args = parser.parse_args()
    run_experiment(args.dataset)
