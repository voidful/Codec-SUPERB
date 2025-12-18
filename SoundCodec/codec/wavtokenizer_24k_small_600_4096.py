from SoundCodec.base_codec.wavtokenizer import WavTokenizerBaseCodec


class Codec(WavTokenizerBaseCodec):
    def config(self):
        self.setting = "wavtokenizer_small_600_24k_4096"
        self.config_url = "https://raw.githubusercontent.com/voidful/WavTokenizer/master/wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        self.ckpt_repo = "novateur/WavTokenizer"
        self.ckpt_filename = "WavTokenizer_small_600_24k_4096.ckpt"
        super().config()
