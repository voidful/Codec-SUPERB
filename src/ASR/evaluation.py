import os
import argparse
import torch
import torchaudio
import whisper
import librosa
import pandas as pd
import jiwer
import editdistance
from pathlib import Path
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

ASR_SAMPLE_RATE = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LibriSpeech(torch.utils.data.Dataset):
    """
    Dataset class to wrap LibriSpeech and trim/pad the audio to a fixed duration, here aiming for 30 seconds.
    Drops the last few seconds if the utterance is slightly longer than 30 seconds.
    """
    def __init__(self, syn_path, n_mels, device=DEVICE):
        # Initialize datasets for both 'test-clean' and 'test-other'
        datasets = []
        for split in ["test-clean", "test-other"]:
            datasets.append(torchaudio.datasets.LIBRISPEECH(
                root=os.path.expanduser("~/.cache"),
                url=split,
                download=True,
            ))
        self.concat_dataset = torch.utils.data.ConcatDataset(datasets)

        # Load synetic files and map them to their IDs
        syn_files = librosa.util.find_files(syn_path)
        self.syn_map = {str(Path(f).stem): f for f in syn_files}
        self.n_mels = n_mels
        self.device = device

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, item):
        # Retrieve reference audio and its metadata
        ref_audio, ref_sample_rate, text, speaker_id, chapter_id, utterance_id = self.concat_dataset[item]
        fileid = f"{speaker_id}-{chapter_id}-{utterance_id:04d}"

        # Retrieve corresponding synetic audio path
        syn_file = self.syn_map.get(fileid)
        syn_audio, syn_sample_rate = torchaudio.load(syn_file)

        # Resample audio to match ASR_SAMPLE_RATE if necessary
        if ref_sample_rate != ASR_SAMPLE_RATE:
            ref_audio = torchaudio.transforms.Resample(ref_sample_rate, ASR_SAMPLE_RATE)(ref_audio)
        if syn_sample_rate != ASR_SAMPLE_RATE:
            syn_audio = torchaudio.transforms.Resample(syn_sample_rate, ASR_SAMPLE_RATE)(syn_audio)

        # Flatten, pad, or trim audio, then compute log Mel spectrograms
        ref_audio = whisper.pad_or_trim(ref_audio.flatten()).to(self.device)
        syn_audio = whisper.pad_or_trim(syn_audio.flatten()).to(self.device)
        ref_mel = whisper.log_mel_spectrogram(ref_audio, self.n_mels)
        syn_mel = whisper.log_mel_spectrogram(syn_audio, self.n_mels)

        return ref_mel, syn_mel, text


def ASR_Eval(syn_path):
    # Initialize Whisper model
    model = whisper.load_model("large")
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    normalizer = EnglishTextNormalizer()

    dataset = LibriSpeech(syn_path, model.dims.n_mels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    ref_hypotheses, syn_hypotheses, references = [], [], []
    for ref_mels, syn_mels, texts in tqdm(loader):
        ref_results = model.decode(ref_mels, options)
        syn_results = model.decode(syn_mels, options)

        ref_hypotheses.extend([result.text for result in ref_results])
        syn_hypotheses.extend([result.text for result in syn_results])
        references.extend(texts)

    # Prepare data and compute WER
    data = pd.DataFrame(dict(ref_hypothesis=ref_hypotheses, syn_hypothesis=syn_hypotheses, reference=references))
    data["ref_hypothesis_clean"] = data["ref_hypothesis"].apply(normalizer.__call__)
    data["syn_hypothesis_clean"] = data["syn_hypothesis"].apply(normalizer.__call__)
    data["reference_clean"] = data["reference"].apply(normalizer.__call__)

    # Calculate Word Error Rate (WER)
    ref_wer = jiwer.wer(list(data["reference_clean"]), list(data["ref_hypothesis_clean"]))
    syn_wer = jiwer.wer(list(data["reference_clean"]), list(data["syn_hypothesis_clean"]))

    print(f"Ref WER: {ref_wer * 100:.2f}%")
    print(f"Syn WER: {syn_wer * 100:.2f}%")

    # Calculate Edit Distance for each pair and add to dataframe
    data["edit_distance_ref"] = [
        editdistance.eval(ref.split(), truth.split())
        for ref, truth in zip(data["ref_hypothesis_clean"], data["reference_clean"])
    ]
    data["edit_distance_syn"] = [
        editdistance.eval(syn.split(), truth.split())
        for syn, truth in zip(data["syn_hypothesis_clean"], data["reference_clean"])
    ]

    avg_edit_distance_ref = data["edit_distance_ref"].mean()
    avg_edit_distance_syn = data["edit_distance_syn"].mean()
    print(f"Ref Edit Distance: {avg_edit_distance_ref:.2f}")
    print(f"Syn Edit Distance: {avg_edit_distance_syn:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run automatic speech recognition experiments.')
    parser.add_argument('--syn_path', type=str, help='Directory containing synetic audio files')
    args = parser.parse_args()
    ASR_Eval(args.syn_path)

