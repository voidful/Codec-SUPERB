import codec.general
import codec.encodec_hf


def load_codec(codec_name):
    module = __import__(f"codec.{codec_name}", fromlist=[codec_name])
    return module.Codec()
