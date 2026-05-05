from SoundCodec.base_codec.llmcodec import LLMCodecBaseCodec

class Codec(LLMCodecBaseCodec):
    def config(self):
        self.setting = "llmcodec_abl_k3"
        self.ckpt_repo = "voidful/llmcodec-abl-k3"
        self.ckpt_filename = "llm-codec.pt"
        super().config()
