from datasets import load_dataset


def load_data():
    cleaned_dataset = load_dataset("DynamicSuperb/IntentClassification_FluentSpeechCommands", "test")
    cleaned_dataset = cleaned_dataset.remove_columns(
        ['file', 'speakerId', 'transcription', 'action', 'object', 'location', 'instruction'])
    return cleaned_dataset
