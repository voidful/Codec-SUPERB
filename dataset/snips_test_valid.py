from datasets import load_dataset


def load_data():
    cleaned_dataset = load_dataset("Codec-SUPERB/SNIPS", split="test+valid")
    return cleaned_dataset
