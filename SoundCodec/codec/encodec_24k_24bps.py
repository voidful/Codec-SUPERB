from SoundCodec.base_codec.encodec import EncodecBaseCodec

class Codec(EncodecBaseCodec):
    def config(self):
        super().config()
        self.model.set_target_bandwidth(24.0)
        self.setting = "encodec_24khz_24"
        self.sampling_rate = 24_000
