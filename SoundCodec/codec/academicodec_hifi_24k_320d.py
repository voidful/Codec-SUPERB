import json
import nlp2
from SoundCodec.base_codec.academicodec import AcademicCodecBaseCodec


class Codec(AcademicCodecBaseCodec):
    def config(self):
        self.setting = "academicodec_hifi_24k_320d"
        nlp2.download_file(
            'https://raw.githubusercontent.com/yangdongchao/AcademiCodec/master/egs/HiFi-Codec-24k-320d/config_24k_320d.json',
            'external_codecs/academicodec_hifi')
        self.config_path = "external_codecs/academicodec_hifi/config_24k_320d.json"
        nlp2.download_file(
            'https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-24k-320d',
            "external_codecs/academicodec_hifi")
        self.ckpt_path = "external_codecs/academicodec_hifi/HiFi-Codec-24k-320d"

        with open(self.config_path, 'r') as f:
            config = json.load(f)
            self.sampling_rate = config['sampling_rate']
