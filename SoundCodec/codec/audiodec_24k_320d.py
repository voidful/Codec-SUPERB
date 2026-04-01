from SoundCodec.base_codec.audiodec import AudioDecBaseCodec
import nlp2
import os
from pathlib import Path


class Codec(AudioDecBaseCodec):
    def config(self):
        self.setting = "audiodec_24k_320d"
        try:
            from AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model
        except:
            raise Exception("Please install AudioDec first. pip install git+https://github.com/voidful/AudioDec.git")
        
        # Get the package root directory (Codec-SUPERB root)
        package_root = Path(__file__).parent.parent.parent.resolve()
        external_codecs_dir = package_root / "external_codecs"
        
        encoder_dir = str(external_codecs_dir / "audiodec_autoencoder_24k_320d")
        decoder_dir = str(external_codecs_dir / "audiodec_vocoder_24k_320d")
        
        # download encoder
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/checkpoint-500000steps.pkl',
            encoder_dir)
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/config.yml',
            encoder_dir)
        self.encoder_config_path = os.path.join(encoder_dir, "checkpoint-500000steps.pkl")
        
        # download decoder
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/checkpoint-500000steps.pkl',
            decoder_dir)
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/config.yml',
            decoder_dir)
        nlp2.download_file(
            "https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/symAD_libritts_24000_hop300_clean.npy",
            decoder_dir
        )
        self.decoder_config_path = os.path.join(decoder_dir, "checkpoint-500000steps.pkl")
        
        self.sampling_rate = 24000
        audiodec_model = AudioDecModel(tx_device=self.device, rx_device=self.device)
        audiodec_model.load_transmitter(self.encoder_config_path)
        audiodec_model.load_receiver(self.encoder_config_path, self.decoder_config_path)
        self.model = audiodec_model
