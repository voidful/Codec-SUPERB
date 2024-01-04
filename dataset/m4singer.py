from .crema_d import load_data as load_data_hear

def load_data():
    return load_data_hear(data_path="/mnt/data/ycevan/svc/dataset/m4singer_processed", split="train", writer_batch_size=2450)