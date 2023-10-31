from datasets import load_dataset


def load_data():
    dataset = load_dataset("mozilla-foundation/common_voice_13_0", "zh-TW", split="test")
    def map_file_to_id(data):
        data['id'] = "".join(data['path'].split("/")[-1:])
        return data
    dataset = dataset.map(map_file_to_id)
    dataset = dataset.remove_columns(
        ['client_id','path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'])
    return dataset
