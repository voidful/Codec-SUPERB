from SoundCodec.base_codec.wavtokenizer import BaseCodec
import nlp2

class Codec(BaseCodec):
    def config(self):
        nlp2.download_file(
            'https://github.com/voidful/WavTokenizer/raw/refs/heads/main/wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
            'wavtokenizer_model')
        self.config_path = "wavtokenizer_model/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

        nlp2.download_file(
            'https://huggingface.co/novateur/WavTokenizer-large-unify-40token/resolve/main/wavtokenizer_large_unify_600_24k.ckpt',
            "wavtokenizer_model")
        self.ckpt_path = "wavtokenizer_model/wavtokenizer_large_unify_600_24k.ckpt"