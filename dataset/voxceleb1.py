from datasets import load_dataset


def load_data():
    dataset = load_dataset("AudioDecBenchmark/Voxceleb1_test_original", split='test')
    return dataset
