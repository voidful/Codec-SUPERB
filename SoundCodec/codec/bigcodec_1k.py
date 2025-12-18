from SoundCodec.base_codec.bigcodec import BigCodecBaseCodec

class Codec(BigCodecBaseCodec):
    def config(self):
        self.ckpt_repo = "Alethia/BigCodec"
        self.ckpt_filename = "bigcodec.pt"
        super().config()
