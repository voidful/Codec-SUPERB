from datasets import Dataset, Audio, DatasetDict
import os
from copy import deepcopy
import subprocess


def load_data(download_path):
    audio_paths = []
    for filename in os.listdir(os.path.join(download_path, 'test'))[:3]:
        if filename.endswith('.mp4'):
            in_file = os.path.join(download_path, 'test', filename)
            out_file = in_file[:-4] + '.wav'
            subprocess.run(f"ffmpeg -i '{in_file}' '{out_file}'; rm {in_file}",shell=True)
            filename = filename[:-4] + '.wav'
        audio_paths.append(os.path.join(download_path, 'test', filename))
    dataset = Dataset.from_dict({"audio": audio_paths, "file": audio_paths}).cast_column("audio", Audio())
    
    def map_file_to_id(data):
        data['id'] = "".join(data['file'].split("/")[-2:])
        return data

    dataset = dataset.map(map_file_to_id)
    dataset = dataset.remove_columns(['file'])
    dataset = DatasetDict({
        "original": deepcopy(dataset),
        "descript_audio_codec": deepcopy(dataset),
        "encodec_hf": deepcopy(dataset),
        "speech_tokenizer": deepcopy(dataset)
    })
    return dataset