from SoundCodec.base_codec.descript_audio_codec import DACBaseCodec


class Codec(DACBaseCodec):
    def config(self):
        self.model_type = "44khz"
        self.sampling_rate = 44_100
