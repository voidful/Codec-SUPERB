from SoundCodec.base_codec.auv import AUVBaseCodec

class Codec(AUVBaseCodec):
    def config(self):
        self.ckpt_repo = "SWivid/AUV"
        self.ckpt_filename = "auv.pt"
        super().config()
