from datasets import load_dataset, Dataset, Audio, Value
from pathlib import Path

from .general import apply_audio_cast


def load_data():

    def map_path_to_id(data):
        path = Path(data['audio']['path'])
        data['id'] = path.name
        return data

    data_dir = "<ESC-50 dir>/audio" # need to be modified
    dataset = load_dataset(data_dir)
    dataset = dataset['train'].map(map_path_to_id)

    return dataset
