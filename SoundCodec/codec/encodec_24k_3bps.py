from SoundCodec.base_codec.encodec import EncodecBaseCodec

class Codec(EncodecBaseCodec):
    def config(self):
        self.model.set_target_bandwidth(3.0)
        self.setting = "encodec_24khz_3"
        self.sampling_rate = 24_000
