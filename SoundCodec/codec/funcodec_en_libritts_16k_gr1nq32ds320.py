import nlp2

from SoundCodec.base_codec.funcodec import FunCodecBaseCodec


class Codec(FunCodecBaseCodec):
    def config(self):
        self.setting = "funcodec_en_libritts-16k-gr1nq32ds320"
        self.sampling_rate = 16000
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/raw/main/config.yaml',
            f"external_codecs/funcodec/{self.setting}")
        self.config_path = f"external_codecs/funcodec/{self.setting}/config.yaml"
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/resolve/main/model.pth',
            f"external_codecs/funcodec/{self.setting}")
        self.ckpt_path = f"external_codecs/funcodec/{self.setting}/model.pth"
        super().config()
