from SoundCodec.base_codec.wavtokenizer import WavTokenizerBaseCodec


class Codec(WavTokenizerBaseCodec):
    def config(self):
        self.setting = "wavtokenizer_large_speech_75token"
        self.config_url = "https://raw.githubusercontent.com/voidful/WavTokenizer/master/wavtokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        self.ckpt_repo = "novateur/WavTokenizer-large-speech-75token"
        self.ckpt_filename = "wavtokenizer_large_speech_320_v2.ckpt"
        super().config()
