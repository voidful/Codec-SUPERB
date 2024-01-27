from datasets import load_dataset


def load_data():
    dataset = load_dataset("DynamicSuperb/SourceSeparation_libri2Mix_test")

    def map_file_to_id(data):
        data['id'] = "".join(data['file'].split("/")[-2:])
        return data

    dataset = dataset.map(map_file_to_id)
    cleaned_dataset = dataset.remove_columns(['file', 's1', 's2', 'instruction'])
    return cleaned_dataset
