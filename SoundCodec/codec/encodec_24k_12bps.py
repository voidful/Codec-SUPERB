from SoundCodec.base_codec.encodec import EncodecBaseCodec

class Codec(EncodecBaseCodec):
    def config(self):
        self.model.set_target_bandwidth(12.0)
        self.setting = "encodec_24khz_12"
        self.sampling_rate = 24_000
