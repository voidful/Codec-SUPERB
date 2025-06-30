import nlp2

from SoundCodec.base_codec.funcodec import FunCodecBaseCodec


class Codec(FunCodecBaseCodec):
    def config(self):
        self.setting = "funcodec_zh_en_general_16k_nq32ds640"
        self.sampling_rate = 16000
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/raw/main/config.yaml',
            f"funcodec/{self.setting}")
        self.config_path = f"funcodec/{self.setting}/config.yaml"
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/resolve/main/model.pth',
            f"funcodec/{self.setting}")
        self.ckpt_path = f"funcodec/{self.setting}/model.pth"
