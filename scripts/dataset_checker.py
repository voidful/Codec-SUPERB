import argparse
import itertools
import numpy as np
from datasets import load_dataset
from SoundCodec.codec import list_codec


def load_datasets(dataset_name, splits):
    print(f"Loading datasets for {dataset_name}...")
    return {split: load_dataset(dataset_name, split=split, streaming=True).shuffle(seed=42, buffer_size=10_000) for
            split in splits}


def sample_data(dataset, sample_size):
    print("Sampling data...")
    reservoir = []
    for item in itertools.islice(dataset, sample_size):
        reservoir.append(item)
    print(f"Sampled {sample_size} items.")
    return reservoir


# Function to check audio differences in a dataset split
def check_audio_differences(dataset, sample_size):
    print("Checking for audio differences...")
    sampled_data = sample_data(dataset, sample_size)
    ids = [item['id'] for item in sampled_data]
    unique_ids = set(ids)

    if len(ids) != len(unique_ids):
        print("Duplicate IDs found in the sampled data.")
        return False

    for u_id in unique_ids:
        audios = [item['audio']['array'] for item in sampled_data if item['id'] == u_id]
        for i in range(len(audios)):
            for j in range(i + 1, len(audios)):
                if np.allclose(audios[i], audios[j]):
                    print(f"Duplicate audios found for ID: {u_id}")
                    return False

    print("No duplicate audios found.")
    return True


# Main execution flow
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check a dataset.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset to check.')
    parser.add_argument('--sample', type=int, help='The number of samples to check.', default=30)
    parser.add_argument('--splits', nargs='+', help='List of splits to check.',
                        default=list_codec())
    args = parser.parse_args()

    datasets = load_datasets(args.dataset, args.splits)
    for split in args.splits:
        print(f"Processing split: {split}")
        are_audios_different = check_audio_differences(datasets[split], args.sample)
        print()
    print("Finished processing all splits.")
