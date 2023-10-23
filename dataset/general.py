from datasets import Audio


def apply_audio_cast(dataset, sampling_rate):
    return dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
