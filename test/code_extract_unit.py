import torch
import torchaudio

from codec import list_codec, load_codec

if __name__ == '__main__':
    for sample_file in ['sample1_16k.wav', 'sample2_22k.wav', 'sample3_48k.wav', 'sample4_16k.wav',
                        'sample5_16k.wav', 'sample6_48k.wav', 'sample7_16k.wav', 'sample8_16k.wav',
                        'sample9_48k.wav', 'sample10_16k.wav']:
        for codec_name in list_codec():
            print(f"Synthesizing {sample_file} with {codec_name}")
            codec = load_codec(codec_name)
            codec_sampling_rate = codec.sampling_rate
            waveform, sample_rate = torchaudio.load(sample_file)
            resampled_waveform = waveform.numpy()[-1]
            data_item = {'audio': {'array': resampled_waveform,
                                   'sampling_rate': sample_rate}}
            audio_array = codec.extract_unit(data_item).unit
            # check all unit are in the same shape or not
            print('audio_array.shape:', audio_array.shape)
            # check dtype is pytorch tensor or not
            assert isinstance(audio_array, torch.Tensor)
            # check is torch array is 2d shape or not, if not raise error
            assert len(audio_array.shape) == 2
        print("=====================================")
