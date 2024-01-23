import torch
from codec.general import save_audio


class BaseCodec:
    def __init__(self):
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            self.EncodecModel = EncodecModel
            self.convert_audio = convert_audio
        except:
            raise Exception("Please install encodec first. pip install encodec")
        self.config()

    def config(self):
        self.model = self.EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)

    @torch.no_grad()
    def synth(self, data, local_save=True):
        encoded_frames = self.extract_unit(data, return_unit_only=False)
        audio_values = \
            self.model.decode(encoded_frames)[0]
        # trim the audio to the same length as the input
        audio_values = audio_values[:, :data['audio']['array'].shape[0]]
        if local_save:
            audio_path = f"dummy_{self.pretrained_model_name}/{data['id']}.wav"
            save_audio(audio_values[0].cpu(), audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values[0].cpu().numpy()
        return data

    @torch.no_grad()
    def extract_unit(self, data, return_unit_only=True):
        wav, sr = data["audio"]["array"], data["audio"]["sampling_rate"]
        # unsqueeze to [B, T], if no batch, B=1
        wav = torch.tensor(wav).unsqueeze(0)
        sr = torch.tensor(sr).unsqueeze(0)
        wav = self.convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [B, n_q, T]
        if return_unit_only:
            return codes
        return encoded_frames
