import torch
from SoundCodec.base_codec.general import save_audio, ExtractedUnit


class BaseCodec:
    def __init__(self):
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.EncodecModel = EncodecModel
            self.convert_audio = convert_audio
            self.model = self.EncodecModel.encodec_model_24khz().to(self.device)
        except:
            raise Exception("Please install encodec first. pip install encodec")
        self.config()

    def config(self):
        self.model = self.EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)
        self.setting = "encodec_24khz_6.0"
        self.sampling_rate = 24_000

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy_{self.setting}/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        wav, sr = data["audio"]["array"], data["audio"]["sampling_rate"]
        # unsqueeze to [B, T], if no batch, B=1
        wav = torch.tensor(wav).unsqueeze(0)
        wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32).to(self.device)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [B, n_q, T]
        return ExtractedUnit(
            unit=codes,
            stuff_for_synth=(encoded_frames, data['audio']['array'].shape[0])
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        encoded_frames, original_shape = stuff_for_synth
        audio_values = \
            self.model.decode(encoded_frames)[0]
        # trim the audio to the same length as the input
        audio_values = audio_values[:, :original_shape].cpu().numpy()
        return audio_values
