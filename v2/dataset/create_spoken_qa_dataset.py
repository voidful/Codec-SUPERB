from SoundCodec import codec
import datasets


# Load and store codecs in a dictionary
def load_codecs():
    codec_dict = {}
    for code in codec.list_codec():
        c = codec.load_codec(code)
        codec_dict[code] = c
    return codec_dict


# Load dataset and rename the 'input_audio' column to 'audio'
def load_dataset():
    dataset = datasets.load_dataset("GSQA/spoken-alpaca-gpt4")
    return dataset.rename_column("input_audio", "audio")


# Filter audio samples based on duration in seconds (<= 7 seconds)
def filter_by_duration(dataitem, sample_rate=22050, max_duration=7):
    duration = len(dataitem['audio']['array']) / sample_rate
    return duration <= max_duration


# Apply codec to extract units from audio and add them to the dataset
def extract_units(item, codec_dict):
    for codec_name, codec_instance in codec_dict.items():
        item[f'{codec_name}_unit'] = codec_instance.extract_unit(item).unit.cpu().numpy()
    return item


# Main processing pipeline
def main():
    codec_dict = load_codecs()

    # Load and filter the dataset
    dataset = load_dataset()
    filtered_dataset = dataset.filter(lambda x: filter_by_duration(x))

    # Map extracted units to the dataset
    processed_dataset = filtered_dataset.map(lambda x: extract_units(x, codec_dict))

    # Split the dataset into training and testing sets
    split_dataset = processed_dataset['train'].train_test_split(test_size=0.2)

    # Push the processed dataset to the Hugging Face Hub
    split_dataset.push_to_hub("voidful/spoken-alpaca-gpt4-unit")


if __name__ == "__main__":
    main()
