from datasets import load_dataset, Dataset, Audio, Value
from pathlib import Path

from .general import apply_audio_cast


def load_data():
    cleaned_dataset = load_dataset("Codec-SUPERB/quesst14", split="audio+dev_queries+eval_queries")
    return cleaned_dataset
