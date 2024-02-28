from SoundCodec.base_codec.general import save_audio, ExtractedUnit
import torch
from audiotools import AudioSignal


class BaseCodec:
    def __init__(self):
        # Reference: https://github.com/descriptinc/descript-audio-codec
        try:
            import dac
        except:
            raise Exception(
                "Please install descript-audio-codec first. pip install git+https://github.com/voidful/descript-audio-codec.git")

        self.config()
        self.model_path = dac.utils.download(model_type=self.model_type)
        self.model = dac.DAC.load(self.model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.sampling_rate = self.sampling_rate

    def config(self):
        self.model_type = "44khz"
        self.sampling_rate = 44100

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        compressed_audio, unit_only = extracted_unit.stuff_for_synth
        data['unit'] = extracted_unit.unit
        decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
        if local_save:
            audio_path = f"dummy-descript-audio-codec-{self.model_type}/{data['id']}.wav"
            save_audio(decompressed_audio, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = decompressed_audio.cpu().numpy()
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        audio_signal = AudioSignal(data["audio"]['array'], data["audio"]['sampling_rate'])
        compressed_audio = self.model.compress(audio_signal, win_duration=5)
        codes = compressed_audio.codes.squeeze(0)
        return ExtractedUnit(
            unit=codes,
            stuff_for_synth=(compressed_audio, codes)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        compressed_audio, unit_only = stuff_for_synth
        decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
        return decompressed_audio.cpu().numpy()
