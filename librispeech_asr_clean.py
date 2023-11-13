import argparse
from datasets import DatasetDict, Audio

import dataset
from codec import list_codec, load_codec
from datasets import load_dataset

dataset_name = "librispeech_asr"

def extract_unit(data, extract_unit_class):
    unit_array = extract_unit_class.extract_unit(data).cpu().numpy()
    data['unit'] = unit_array
    return data

def apply_audio_cast(dataset, sampling_rate):
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


cleaned_dataset = load_dataset(dataset_name, "clean", split='train.100')
sampling_rate = cleaned_dataset[0]['audio']['sampling_rate']
datasets_dict = DatasetDict()
for codec_name in list_codec():
    print(f"Synthesizing dataset with {codec_name}")
    codec = load_codec(codec_name)
    cleaned_dataset_with_audio = apply_audio_cast(cleaned_dataset, codec.sampling_rate)
    synthesized_dataset = cleaned_dataset_with_audio.map(extract_unit, fn_kwargs={'extract_unit_class': codec})
    synthesized_dataset = synthesized_dataset.remove_columns(['file', 'audio', 'speaker_id', 'chapter_id'])
    datasets_dict[f'{codec_name}'] = synthesized_dataset

datasets_dict.push_to_hub(f"AudioDecBenchmark/{dataset_name}")
