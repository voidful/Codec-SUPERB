from datasets import load_dataset


def load_data():
    cleaned_dataset = load_dataset("librispeech_asr", "all", split="test.clean+test.other")
    cleaned_dataset = cleaned_dataset.remove_columns(['file', 'speaker_id', 'chapter_id'])
    return cleaned_dataset
