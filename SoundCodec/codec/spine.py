from SoundCodec.base_codec.spine import SpineBaseCodec


class Codec(SpineBaseCodec):
    def config(self):
        super().config()
        self.setting = "spine_24k"
