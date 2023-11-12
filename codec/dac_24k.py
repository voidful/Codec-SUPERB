from codec.base_codec.descript_audio_codec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.model_type = "24khz"
        self.sampling_rate = 24_000
