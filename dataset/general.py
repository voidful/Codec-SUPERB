from datasets import Audio


def apply_audio_cast(dataset, sampling_rate):
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))


def extract_unit(data, extract_unit_class):
    unit_array = extract_unit_class.extract_unit(data).cpu().numpy()
    data['unit'] = unit_array
    return data
