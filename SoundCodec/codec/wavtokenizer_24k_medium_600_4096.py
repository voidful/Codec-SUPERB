from SoundCodec.base_codec.wavtokenizer import WavTokenizerBaseCodec


class Codec(WavTokenizerBaseCodec):
    def config(self):
        self.setting = "wavtokenizer_medium_speech_75token"
        self.config_repo = "novateur/WavTokenizer-medium-speech-75token"
        self.config_filename = "wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        self.ckpt_repo = "novateur/WavTokenizer-medium-speech-75token"
        self.ckpt_filename = "wavtokenizer_medium_speech_320_24k_v2.ckpt"
        super().config()
