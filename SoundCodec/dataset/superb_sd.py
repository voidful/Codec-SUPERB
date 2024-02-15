from datasets import load_dataset


def load_data():
    dataset = load_dataset("superb", 'sd', split="test", streaming=True)
    dataset = dataset.rename_column('record_id', 'id')
    dataset = dataset.remove_columns(['file', 'start', 'end', 'speakers'])
    return dataset
