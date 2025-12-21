from SoundCodec.base_codec.speech_tokenizer import SpeechTokenizerBaseCodec
import nlp2

class Codec(SpeechTokenizerBaseCodec):
    def config(self):
        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/raw/main/speechtokenizer_hubert_avg/config.json',
            'external_codecs/speechtokenizer_hubert_avg')
        self.config_path = "external_codecs/speechtokenizer_hubert_avg/config.json"

        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
            "external_codecs/speechtokenizer_hubert_avg")
        self.ckpt_path = "external_codecs/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
