from SoundCodec.base_codec.descript_audio_codec import DACBaseCodec


class Codec(DACBaseCodec):
    def config(self):
        self.model_type = "24khz"
        self.sampling_rate = 24_000
