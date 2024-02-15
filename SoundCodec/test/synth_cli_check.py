import subprocess

import torch

from benchmarking import compute_metrics
from SoundCodec.codec import load_codec
import torchaudio
import numpy as np


def test_codec(filename, codec_cli_operation, codec_python_operation, codec_sampling_rate):
    waveform, sample_rate = torchaudio.load(filename)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=codec_sampling_rate)
    resampled_waveform = resampler(waveform)
    resampled_waveform = resampled_waveform.numpy()[-1]
    codec_audio_array = codec_python_operation(filename, codec_sampling_rate)
    cli_audio_array = codec_cli_operation(filename, codec_sampling_rate)

    # compare the two arrays
    print("Ours", codec_audio_array, codec_audio_array.shape)
    print("CLI", cli_audio_array, cli_audio_array.shape)
    print("diff mean:", np.mean(np.abs(codec_audio_array - cli_audio_array)))
    print("diff std:", np.std(np.abs(codec_audio_array - cli_audio_array)))
    print("diff max:", np.max(np.abs(codec_audio_array - cli_audio_array)))
    print("diff min:", np.min(np.abs(codec_audio_array - cli_audio_array)))
    print("diff median:", np.median(np.abs(codec_audio_array - cli_audio_array)))
    print("is close?", np.mean(np.abs(codec_audio_array - cli_audio_array)) < 0.0001)

    # read filename audio to numpy and convert to 24000
    print("CLI to Ori", compute_metrics({'audio': {'array': resampled_waveform, 'sampling_rate': codec_sampling_rate}},
                                        {'audio': {'array': cli_audio_array, 'sampling_rate': codec_sampling_rate}},
                                        120))
    print("Our to Ori", compute_metrics({'audio': {'array': resampled_waveform, 'sampling_rate': codec_sampling_rate}},
                                        {'audio': {'array': codec_audio_array, 'sampling_rate': codec_sampling_rate}},
                                        120))
    return np.mean(np.abs(codec_audio_array - cli_audio_array)) < 0.0001


def encodec_cli_operation(filename, codec_sampling_rate=24000):
    output_filename = filename.replace('.wav', '_encodec_cli.wav')
    command = ['encodec', '-f', '-r', '-b', '6.0', filename, output_filename]
    subprocess.run(command, check=True)
    waveform, sample_rate = torchaudio.load(output_filename)
    return waveform.numpy()[-1]


def encodec_python_operation(filename, codec_sampling_rate=24000):
    codec = load_codec('encodec_24k_6bps')
    waveform, sample_rate = torchaudio.load(filename)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=codec_sampling_rate)
    resampled_waveform = resampler(waveform)
    resampled_waveform = resampled_waveform.numpy()[-1]
    data_item = {'audio': {'array': resampled_waveform,
                           'sampling_rate': codec_sampling_rate}}  # use the new sampling rate
    audio_array = codec.synth(data_item, local_save=False)['audio']['array']
    # save audio_array to wav in local
    filename = filename.split('/')[-1].split('.')[0]
    output_filename = filename + '_encodec_our.wav'
    torchaudio.save(output_filename, torch.tensor(np.array(audio_array)),
                    codec_sampling_rate,
                    encoding='PCM_S',
                    bits_per_sample=16)
    waveform, sample_rate = torchaudio.load(output_filename)
    return waveform.numpy()[-1]


def dac_cli_operation(filename, codec_sampling_rate=16000):
    output_filename = filename.replace('.wav', '_dac_cli')
    # get device cuda or cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    command = ['python', '-m', 'dac', 'encode', filename, '--output', output_filename, '--device', device,
               '--model_type', "16khz"]
    subprocess.run(command, check=True)
    filename = filename.split('/')[-1].split('.')[0]

    dac_decode_filename = output_filename + '/' + filename + '.dac'
    command = ['python', '-m', 'dac', 'decode', dac_decode_filename,
               '--output', output_filename, '--device', device,
               '--model_type', "16khz"]
    subprocess.run(command, check=True)
    waveform, sample_rate = torchaudio.load(output_filename + "/" + filename + '.wav')
    return waveform.numpy()[-1]


def dac_16k_python_operation(filename, codec_sampling_rate=16000):
    codec = load_codec('dac_16k')
    waveform, sample_rate = torchaudio.load(filename)
    resampled_waveform = waveform.numpy()[-1]
    data_item = {'audio': {'array': resampled_waveform,
                           'sampling_rate': sample_rate}}  # use the new sampling rate
    audio_array = codec.synth(data_item, local_save=False)['audio']['array']
    # save audio_array to wav in local
    filename = filename.split('/')[-1].split('.')[0]
    output_filename = filename + '_dac_our.wav'
    torchaudio.save(output_filename, torch.tensor(np.array(audio_array)),
                    codec_sampling_rate,
                    encoding='PCM_S',
                    bits_per_sample=16)
    waveform, sample_rate = torchaudio.load(output_filename)
    return waveform.numpy()[-1]


def dac_22k_python_operation(filename, codec_sampling_rate=16000):
    codec = load_codec('dac_16k')
    waveform, sample_rate = torchaudio.load(filename)
    resampled_waveform = waveform.numpy()[-1]
    data_item = {'audio': {'array': resampled_waveform,
                           'sampling_rate': sample_rate}}  # use the new sampling rate
    audio_array = codec.synth(data_item, local_save=False)['audio']['array']
    # save audio_array to wav in local
    filename = filename.split('/')[-1].split('.')[0]
    output_filename = filename + '_dac_our.wav'
    torchaudio.save(output_filename, torch.tensor(np.array(audio_array)),
                    codec_sampling_rate,
                    encoding='PCM_S',
                    bits_per_sample=16)
    waveform, sample_rate = torchaudio.load(output_filename)
    return waveform.numpy()[-1]


def dac_48k_python_operation(filename, codec_sampling_rate=16000):
    codec = load_codec('dac_16k')
    waveform, sample_rate = torchaudio.load(filename)
    resampled_waveform = waveform.numpy()[-1]
    data_item = {'audio': {'array': resampled_waveform,
                           'sampling_rate': sample_rate}}  # use the new sampling rate
    audio_array = codec.synth(data_item, local_save=False)['audio']['array']
    # save audio_array to wav in local
    filename = filename.split('/')[-1].split('.')[0]
    output_filename = filename + '_dac_our.wav'
    torchaudio.save(output_filename, torch.tensor(np.array(audio_array)),
                    codec_sampling_rate,
                    encoding='PCM_S',
                    bits_per_sample=16)
    waveform, sample_rate = torchaudio.load(output_filename)
    return waveform.numpy()[-1]


if __name__ == '__main__':
    for sample_file in ['sample1_16k.wav', 'sample2_22k.wav', 'sample3_48k.wav', 'sample4_16k.wav',
                        'sample5_16k.wav', 'sample6_48k.wav', 'sample7_16k.wav', 'sample8_16k.wav',
                        'sample9_48k.wav', 'sample10_16k.wav']:
        print("Checking", sample_file)
        print("encodec")
        test_codec(sample_file, encodec_cli_operation, encodec_python_operation, 24000)
        print("dac 16k")
        test_codec(sample_file, dac_cli_operation, dac_16k_python_operation, 16000)
        print("dac 22k")
        test_codec(sample_file, dac_cli_operation, dac_22k_python_operation, 22000)
        print("dac 48k")
        test_codec(sample_file, dac_cli_operation, dac_48k_python_operation, 48000)
        print("=====================================")
