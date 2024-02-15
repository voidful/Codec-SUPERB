from datasets import load_dataset


def load_data():
    dataset = load_dataset("superb", "ks", split="test")

    def map_file_to_id(data):
        data['id'] = "".join(data['file'].split("/")[-2:])
        return data

    dataset = dataset.map(map_file_to_id)
    dataset = dataset.remove_columns(['label', 'file'])
    return dataset
