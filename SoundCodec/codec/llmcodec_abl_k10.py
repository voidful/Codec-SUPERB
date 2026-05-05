from SoundCodec.base_codec.llmcodec import LLMCodecBaseCodec

class Codec(LLMCodecBaseCodec):
    def config(self):
        self.setting = "llmcodec_abl_k10"
        self.ckpt_repo = "voidful/llmcodec-abl-k10"
        self.ckpt_filename = "llm-codec.pt"
        super().config()
