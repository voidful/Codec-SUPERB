from .crema_d import load_data as load_data_hear

def load_data():
    return load_data_hear(data_path="/mnt/data/ycevan/AudioDecBenchmark/storage/hear-2021.0.6/tasks/processed_vocal_imitation", split="train", writer_batch_size=2500)