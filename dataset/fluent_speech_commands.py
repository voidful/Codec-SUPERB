from datasets import load_dataset


def load_data():
    cleaned_dataset = load_dataset("DynamicSuperb/IntentClassification_FluentSpeechCommands", "test")

    def map_file_to_id(data):
        data['id'] = data['file']
        return data

    cleaned_dataset = cleaned_dataset.map(map_file_to_id)
    cleaned_dataset = cleaned_dataset.remove_columns(
        ['file', 'speakerId', 'transcription', 'action', 'object', 'location', 'instruction'])
    cleaned_dataset = cleaned_dataset['test']
    return cleaned_dataset
