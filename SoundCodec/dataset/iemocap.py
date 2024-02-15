from datasets import load_dataset, concatenate_datasets



def load_data():
    dataset = load_dataset("minoosh/IEMOCAP_Speech_dataset")
    dataset = concatenate_datasets([v for k, v in dataset.items()])

    def map_file_to_id(data):
        data['id'] = data['TURN_NAME']
        return data

    dataset = dataset.map(map_file_to_id)
    dataset = dataset.remove_columns(['emotion', 'TURN_NAME'])
    return dataset
