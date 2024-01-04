from .crema_d import load_data as load_data_hear

def load_data():
    return load_data_hear(data_path="/mnt/data/ycevan/AudioDecBenchmark/storage/musdb18", split="train+test", writer_batch_size=15)

if __name__=='__main__':
    dataset = load_data()
    print(dataset)