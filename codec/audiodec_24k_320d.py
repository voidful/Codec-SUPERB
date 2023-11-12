from base_codec.audiodec import BaseCodec


class Codec(BaseCodec):
    def __init__(self):
        self.device = "cuda"
        self.config()