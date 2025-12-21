from SoundCodec.base_codec.audiodec import AudioDecBaseCodec
import nlp2


class Codec(AudioDecBaseCodec):
    def config(self):
        self.setting = "audiodec_24k_320d"
        try:
            from AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model
        except:
            raise Exception("Please install AudioDec first. pip install git+https://github.com/voidful/AudioDec.git")
        # download encoder
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/checkpoint-500000steps.pkl',
            'external_codecs/audiodec_autoencoder_24k_320d')
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/config.yml',
            "external_codecs/audiodec_autoencoder_24k_320d")
        self.encoder_config_path = "external_codecs/audiodec_autoencoder_24k_320d/checkpoint-500000steps.pkl"
        # download decoder
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/checkpoint-500000steps.pkl',
            'external_codecs/audiodec_vocoder_24k_320d')
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/config.yml',
            "external_codecs/audiodec_vocoder_24k_320d")
        nlp2.download_file(
            "https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/symAD_libritts_24000_hop300_clean.npy",
            "external_codecs/audiodec_vocoder_24k_320d"
        )
        self.decoder_config_path = "external_codecs/audiodec_vocoder_24k_320d/checkpoint-500000steps.pkl"
        self.sampling_rate = 24000
        audiodec_model = AudioDecModel(tx_device=self.device, rx_device=self.device)
        audiodec_model.load_transmitter(self.encoder_config_path)
        audiodec_model.load_receiver(self.encoder_config_path, self.decoder_config_path)
        self.model = audiodec_model
