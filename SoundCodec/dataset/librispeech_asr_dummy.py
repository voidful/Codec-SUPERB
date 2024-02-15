from datasets import load_dataset

def load_data():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    return dataset.remove_columns(['file', 'text', 'speaker_id', 'chapter_id'])
