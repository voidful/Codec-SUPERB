from SoundCodec.base_codec.auv import AUVBaseCodec

# LLMCodec shares the AUV backbone — only the default checkpoint differs, and
# that is supplied by the codec wrapper in SoundCodec/codec/llmcodec.py.
LLMCodecBaseCodec = AUVBaseCodec
