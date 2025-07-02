from SoundCodec.base_codec.unicodec import UnicodecBaseCodec
import nlp2

class Codec(UnicodecBaseCodec):
    def _download_resources(self):
        nlp2.download_file(
            'https://huggingface.co/Yidiii/UniCodec_ckpt/resolve/main/unicode.ckpt',
            'unicodec_model'
        )
        self.ckpt_path = "unicodec_model/unicode.ckpt"

        nlp2.download_file(
            'https://huggingface.co/huseinzol05/UniCodec-mirror/raw/main/config.yaml',
            "unicodec_model"
        )
        self.config_path = "unicodec_model/config.yaml"