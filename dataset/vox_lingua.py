from .crema_d import load_data as load_data_hear

def load_data():
    return load_data_hear(data_path="/mnt/data/ycevan/AudioDecBenchmark/storage/hear-2021.0.6/tasks/vox_lingua_top10-hear2021-full/48000", split="train")