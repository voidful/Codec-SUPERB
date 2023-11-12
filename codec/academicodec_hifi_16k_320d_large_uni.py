import json
import nlp2
from codec.base_codec.academicodec import BaseCodec


class Codec(BaseCodec):
    def config(self):
        self.setting = "academicodec_hifi_16k_320d_large_uni"
        nlp2.download_file(
            'https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-16k-320d/config_16k_320d.json',
            'academicodec_hifi')
        self.config_path = "academicodec_hifi/config_16k_320d.json"
        nlp2.download_file(
            'https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-16k-320d-large-universal',
            "academicodec_hifi")
        self.ckpt_path = "academicodec_hifi/HiFi-Codec-16k-320d-large-universal"
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            self.sampling_rate = config['sampling_rate']
