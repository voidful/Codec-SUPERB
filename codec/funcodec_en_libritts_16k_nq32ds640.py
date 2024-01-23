import nlp2

from base_codec.funcodec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.setting = "funcodec_en_libritts-16k-nq32ds640"
        self.sampling_rate = 16000
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch/raw/main/config.yaml',
            f"funcodec/{self.setting}")
        self.config_path = f"funcodec/{self.setting}/config.yaml"
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch/resolve/main/model.pth',
            f"funcodec/{self.setting}")
        self.ckpt_path = f"funcodec/{self.setting}/model.pth"
