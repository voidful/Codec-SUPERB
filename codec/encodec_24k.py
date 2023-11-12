from codec.base_codec.encodec_hf import BaseCodec

class Codec(BaseCodec):
    def config(self):
        self.pretrained_model_name = "facebook/encodec_24khz"
