import nlp2
import numpy
import torch

from SoundCodec.base_codec.general import save_audio, ExtractedUnit


class BaseCodec:
    def __init__(self):
        try:
            from wavtokenizer.decoder.pretrained import WavTokenizer
        except:
            raise Exception(
                "Please install wavtokenizer first. pip install git+https://github.com/voidful/WavTokenizer.git")

        self.config()
        self.model = WavTokenizer.from_pretrained0802(self.config_path, self.ckpt_path)
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.sampling_rate = 24000

    def config(self):
        nlp2.download_file(
            'https://github.com/voidful/WavTokenizer/raw/refs/heads/main/wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            'wavtokenizer_model')
        self.config_path = "wavtokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.json"

        nlp2.download_file(
            'https://huggingface.co/novateur/WavTokenizer-large-unify-40token/resolve/main/wavtokenizer_large_unify_600_24k.ckpt',
            "wavtokenizer_model")
        self.ckpt_path = "wavtokenizer/wavtokenizer_large_unify_600_24k.ckpt"

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy-WavTokenizer/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(numpy.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        bandwidth_id = torch.tensor([0])
        features, discrete_code = self.model.encode_infer(wav, bandwidth_id=bandwidth_id)
        return ExtractedUnit(
            unit=discrete_code[0],
            stuff_for_synth=(features, bandwidth_id)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        features, bandwidth_id = stuff_for_synth
        audio_out = self.model.decode(features, bandwidth_id=bandwidth_id)
        wav = audio_out.detach().cpu().numpy()
        return wav
