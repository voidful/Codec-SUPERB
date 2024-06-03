import os
import re
import argparse

# Function to calculate average
def calculate_average(data):
    return sum(data) / len(data) if data else 0

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process log files to calculate average metrics.')
parser.add_argument('--log_dir', type=str, required=True, help='Directory where log files are located')
args = parser.parse_args()

# Specify the speech and audio datasets
speech_datasets = {
    'crema_d', 'fluent_speech_commands', 'libri2Mix_test', 'librispeech',
    'quesst', 'snips_test_valid_subset', 'vox_lingua_top10', 'voxceleb1'
}

audio_datasets = {'esc50', 'fsd50k', 'gunshot_triangulation'}

# Initialize dictionaries to store data
speech_data = {
    'SDR': [],
    'Mel_Loss': [],
    'STOI': [],
    'PESQ': []
}

audio_data = {
    'SDR': [],
    'Mel_Loss': []
}

# Regular expression patterns
dataset_pattern = re.compile(r'Codec SUPERB objective metric evaluation on (\w+)')
sdr_pattern = re.compile(r'SDR: mean score is: ([\d.]+)')
mel_loss_pattern = re.compile(r'mel_loss: mean score is: ([\d.]+)')
stoi_pattern = re.compile(r'stoi: mean score is: ([\d.]+)')
pesq_pattern = re.compile(r'pesq: mean score is: ([\d.]+)')

# Iterate over log files directory
for file_name in os.listdir(args.log_dir):
    if file_name.endswith('.log'):
        with open(os.path.join(args.log_dir, file_name), 'r') as file:
            content = file.read()
            dataset_match = dataset_pattern.search(content)
            sdr_match = sdr_pattern.search(content)
            mel_loss_match = mel_loss_pattern.search(content)
            stoi_match = stoi_pattern.search(content)
            pesq_match = pesq_pattern.search(content)

            if dataset_match:
                dataset = dataset_match.group(1)
                if dataset in speech_datasets and sdr_match and mel_loss_match and stoi_match and pesq_match:
                    sdr = float(sdr_match.group(1))
                    mel_loss = float(mel_loss_match.group(1))
                    stoi = float(stoi_match.group(1))
                    pesq = float(pesq_match.group(1))

                    speech_data['SDR'].append(sdr)
                    speech_data['Mel_Loss'].append(mel_loss)
                    speech_data['STOI'].append(stoi)
                    speech_data['PESQ'].append(pesq)

                elif dataset in audio_datasets and sdr_match and mel_loss_match:
                    sdr = float(sdr_match.group(1))
                    mel_loss = float(mel_loss_match.group(1))

                    audio_data['SDR'].append(sdr)
                    audio_data['Mel_Loss'].append(mel_loss)

# Calculate averages
mean_speech_sdr = calculate_average(speech_data['SDR'])
mean_speech_mel_loss = calculate_average(speech_data['Mel_Loss'])
mean_speech_stoi = calculate_average(speech_data['STOI'])
mean_speech_pesq = calculate_average(speech_data['PESQ'])

mean_audio_sdr = calculate_average(audio_data['SDR'])
mean_audio_mel_loss = calculate_average(audio_data['Mel_Loss'])

# Print results
print(f'Average SDR for speech datasets: {mean_speech_sdr}')
print(f'Average Mel_Loss for speech datasets: {mean_speech_mel_loss}')
print(f'Average STOI for speech datasets: {mean_speech_stoi}')
print(f'Average PESQ for speech datasets: {mean_speech_pesq}')

print(f'Average SDR for audio datasets: {mean_audio_sdr}')
print(f'Average Mel_Loss for audio datasets: {mean_audio_mel_loss}')
