from SoundCodec.base_codec.s3tokenizer import S3TokenizerBaseCodec

class Codec(S3TokenizerBaseCodec):
    def config(self):
        self.model_name = "speech_tokenizer_v3_25hz"
        super().config()
