from SoundCodec.base_codec.llmcodec import LLMCodecBaseCodec

class Codec(LLMCodecBaseCodec):
    def config(self):
        self.setting = "llmcodec"
        self.ckpt_repo = "voidful/llm-codec-abl-ftp"
        self.ckpt_filename = "llm-codec.pt"
        super().config()
