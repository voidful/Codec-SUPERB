from codec.general import save_audio

from audiotools import AudioSignal


class BaseCodec:
    def __init__(self, pretrained_model_name="descript-audio-codec"):
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
        compressed_audio = self.extract_unit(data, return_unit_only=False)
        decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
        audio_path = f"dummy-descript-audio-codec-{self.model_type}/{data['id']}.wav"
        save_audio(decompressed_audio, audio_path, self.sampling_rate)
        data['audio'] = audio_path
        return data

    def extract_unit(self, data, return_unit_only=True):
        audio_path = data["audio"]["path"]
        audio_signal = AudioSignal(audio_path)

        if audio_signal.sample_rate != self.sampling_rate:
            audio_signal.resample(self.sampling_rate)

        compressed_audio = self.model.compress(audio_signal)
        if return_unit_only:
            return compressed_audio.codes.squeeze(0)
        return compressed_audio
