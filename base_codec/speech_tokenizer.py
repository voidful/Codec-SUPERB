from codec.general import save_audio
import torchaudio
import torch
import nlp2
from pathlib import Path


class BaseCodec:
    def __init__(self):
        try:
            from speechtokenizer import SpeechTokenizer
        except:
            raise Exception("Please install SpeechTokenizer first. pip install -U speechtokenizer")

        self.config()
        self.model = SpeechTokenizer.load_from_checkpoint(self.config_path, self.ckpt_path)
        self.model.eval()
        self.model = self.model.to('cuda')
        self.sampling_rate = self.model.sample_rate

    def config(self):
        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/raw/main/speechtokenizer_hubert_avg/config.json',
            'speechtokenizer_hubert_avg')
        self.config_path = "speechtokenizer_hubert_avg/config.json"

        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
            "speechtokenizer_hubert_avg")
        self.ckpt_path = "speechtokenizer_hubert_avg/SpeechTokenizer.pt"

    def synth(self, data):
        audio_path = f"dummy-SpeechTokenizer/{data['id']}.wav"
        with torch.no_grad():
            if not Path(audio_path).exists():
                codes = self.extract_unit(data, return_unit_only=False)
                RVQ_1 = codes[:1, :, :]  # Contain content info, can be considered as semantic tokens
                RVQ_supplement = codes[1:, :, :]  # Contain timbre info, complete info lost by the first quantizer
                # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
                wav = self.model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0).to('cuda'))
                wav = wav.detach().cpu().squeeze(0)
                save_audio(wav, audio_path, self.sampling_rate)
            data['audio'] = audio_path
            return data

    def extract_unit(self, data, return_unit_only=True):
        with torch.no_grad():
            wav = torch.tensor([data["audio"]['array']], dtype=torch.float32).to('cuda')
            sampling_rate = data["audio"]['sampling_rate']
            if sampling_rate != self.sampling_rate:
                wav = torchaudio.functional.resample(wav, sampling_rate, self.sampling_rate)
            wav = wav.unsqueeze(0)
            codes = self.model.encode(wav.to('cuda'))
            if return_unit_only:
                # swap dim 0 and 1, and squeeze dim 0
                return codes.permute(1, 0, 2).squeeze(0)
            return codes
