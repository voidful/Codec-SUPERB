from datasets import load_dataset


def load_data():
    cleaned_dataset = load_dataset("librispeech_asr", "clean")
    cleaned_dataset = cleaned_dataset.remove_columns(['file', 'speaker_id', 'chapter_id'])
    return cleaned_dataset
