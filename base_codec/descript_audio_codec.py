from codec.general import save_audio
import torch
from audiotools import AudioSignal
from pathlib import Path

class BaseCodec:
    def __init__(self):
        # Reference: https://github.com/descriptinc/descript-audio-codec
        try:
            import dac
        except:
            raise Exception("Please install descript-audio-codec first. pip install descript-audio-codec")

        self.config()
        self.model_path = dac.utils.download(model_type=self.model_type)
        self.model = dac.DAC.load(self.model_path)
        self.device = "cuda"
        self.model.to(self.device)
        self.sampling_rate = self.sampling_rate

    def config(self):
        self.model_type = "44khz"
        self.sampling_rate = 44100

    def synth(self, data):
        with torch.no_grad():
            audio_path = f"dummy-descript-audio-codec-{self.model_type}/{data['id']}.wav"
            if not Path(audio_path).exists():
                compressed_audio = self.extract_unit(data, return_unit_only=False)
                decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
                save_audio(decompressed_audio, audio_path, self.sampling_rate)
            data['audio'] = audio_path
            return data

    def extract_unit(self, data, return_unit_only=True):
        with torch.no_grad():
            audio_signal = AudioSignal(data["audio"]['array'], data["audio"]['sampling_rate'])
            if audio_signal.sample_rate != self.sampling_rate:
                audio_signal.resample(self.sampling_rate)
            compressed_audio = self.model.compress(audio_signal)
            if return_unit_only:
                return compressed_audio.codes.squeeze(0)
            return compressed_audio
