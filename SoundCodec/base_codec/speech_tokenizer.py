import numpy

from SoundCodec.base_codec.general import save_audio, ExtractedUnit
import torchaudio
import torch
import nlp2


class BaseCodec:
    def __init__(self):
        try:
            from speechtokenizer import SpeechTokenizer
        except:
            raise Exception("Please install SpeechTokenizer first. pip install -U speechtokenizer")

        self.config()
        self.model = SpeechTokenizer.load_from_checkpoint(self.config_path, self.ckpt_path)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
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

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy-SpeechTokenizer/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(numpy.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        wav = wav.unsqueeze(0)
        codes = self.model.encode(wav.to(self.device))
        return ExtractedUnit(
            unit=codes.permute(1, 0, 2).squeeze(0),
            stuff_for_synth=codes
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        codes = stuff_for_synth
        RVQ_1 = codes[:1, :, :]  # Contain content info, can be considered as semantic tokens
        RVQ_supplement = codes[1:, :, :]  # Contain timbre info, complete info lost by the first quantizer
        # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
        wav = self.model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0).to(self.device))
        wav = wav.detach().cpu().squeeze(0).numpy()
        return wav
