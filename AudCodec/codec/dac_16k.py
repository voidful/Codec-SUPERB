from AudCodec.base_codec.descript_audio_codec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.model_type = "16khz"
        self.sampling_rate = 16_000
