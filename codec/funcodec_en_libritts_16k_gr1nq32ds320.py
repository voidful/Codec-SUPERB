import nlp2

from codec.base_codec.funcodec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.setting = "funcodec_en_libritts-16k-gr1nq32ds320"
        self.sampling_rate = 16000
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/raw/main/config.yaml',
            self.setting)
        self.config_path = f"{self.setting}/config.yaml"
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch/resolve/main/model.pth',
            self.setting)
        self.ckpt_path = f"{self.setting}/model.pth"
