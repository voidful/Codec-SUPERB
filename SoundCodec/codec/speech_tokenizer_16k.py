from SoundCodec.base_codec.speech_tokenizer import SpeechTokenizerBaseCodec
import nlp2

class Codec(SpeechTokenizerBaseCodec):
    def config(self):
        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/raw/main/speechtokenizer_hubert_avg/config.json',
            'speechtokenizer_hubert_avg')
        self.config_path = "speechtokenizer_hubert_avg/config.json"

        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
            "speechtokenizer_hubert_avg")
        self.ckpt_path = "speechtokenizer_hubert_avg/SpeechTokenizer.pt"
