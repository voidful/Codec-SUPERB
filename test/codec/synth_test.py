import subprocess

import torch

from benchmarking import compute_metrics
from codec import load_codec
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
    print("Ours", codec_audio_array)
    print("CLI", cli_audio_array)
    print("diff mean:", np.mean(np.abs(codec_audio_array - cli_audio_array)))
    print("diff std:", np.std(np.abs(codec_audio_array - cli_audio_array)))
    print("diff max:", np.max(np.abs(codec_audio_array - cli_audio_array)))
    print("diff min:", np.min(np.abs(codec_audio_array - cli_audio_array)))
    print("diff median:", np.median(np.abs(codec_audio_array - cli_audio_array)))
    print("is close?", np.mean(np.abs(codec_audio_array - cli_audio_array)) < 0.0001)

    # read filename audio to numpy and convert to 24000
    print("CLI to Ori",compute_metrics({'audio': {'array': resampled_waveform, 'sampling_rate': codec_sampling_rate}},
                          {'audio': {'array': cli_audio_array, 'sampling_rate': codec_sampling_rate}}, 120))
    print("Our to Ori",compute_metrics({'audio': {'array': resampled_waveform, 'sampling_rate': codec_sampling_rate}},
                          {'audio': {'array': codec_audio_array, 'sampling_rate': codec_sampling_rate}}, 120))
    return np.allclose(codec_audio_array, cli_audio_array)


def encodec_cli_operation(filename, codec_sampling_rate=24000):
    output_filename = filename.replace('.wav', '_encodec_cli.wav')
    command = ['encodec', '-f', '-r', '-b', '6.0', filename, output_filename]
    subprocess.run(command, check=True)
    waveform, sample_rate = torchaudio.load(output_filename)
    return waveform.numpy()[-1]


def encodec_python_operation(filename, codec_sampling_rate=24000):
    codec = load_codec('encodec_24k')
    waveform, sample_rate = torchaudio.load(filename)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=codec_sampling_rate)
    resampled_waveform = resampler(waveform)
    resampled_waveform = resampled_waveform.numpy()[-1]
    # resampled_waveform = waveform.numpy()[-1]
    data_item = {'audio': {'array': resampled_waveform,
                           'sampling_rate': codec_sampling_rate}}  # use the new sampling rate
    audio_array = codec.synth(data_item, save_audio=False)['audio']['array']
    # save audio_array to wav in local
    filename = filename.split('/')[-1].split('.')[0]
    torchaudio.save(filename + 'encodec_our.wav', torch.tensor(np.array([audio_array])),
                    codec_sampling_rate,
                    encoding='PCM_S',
                    bits_per_sample=16)
    return audio_array


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


def dac_python_operation(filename, codec_sampling_rate=16000):
    codec = load_codec('dac_16k')
    waveform, sample_rate = torchaudio.load(filename)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=codec_sampling_rate)
    resampled_waveform = resampler(waveform)
    resampled_waveform = resampled_waveform.numpy()[-1]
    # resampled_waveform = waveform.numpy()[-1]
    data_item = {'audio': {'array': resampled_waveform,
                           'sampling_rate': codec_sampling_rate}}  # use the new sampling rate
    audio_array = codec.synth(data_item, save_audio=False)['audio']['array']
    # save audio_array to wav in local
    filename = filename.split('/')[-1].split('.')[0]
    torchaudio.save(filename + 'dac_our.wav', torch.tensor(np.array([audio_array])),
                    codec_sampling_rate,
                    encoding='PCM_S',
                    bits_per_sample=16)
    return audio_array


if __name__ == '__main__':
    test_codec('./sample_16k.wav', encodec_cli_operation, encodec_python_operation, 24000)
    test_codec('./sample_16k.wav', dac_cli_operation, dac_python_operation, 16000)
