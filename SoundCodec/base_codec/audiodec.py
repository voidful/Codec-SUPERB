import nlp2
import torch
from SoundCodec.base_codec.general import save_audio, ExtractedUnit


class BaseCodec:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config()

    def config(self):
        self.setting = "audiodec_24k_320d"
        try:
            from AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model
        except:
            raise Exception("Please install AudioDec first. pip install git+https://github.com/voidful/AudioDec.git")
        # download encoder
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/checkpoint-500000steps.pkl',
            'audiodec_autoencoder_24k_320d')
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/autoencoder/symAD_libritts_24000_hop300/config.yml',
            "audiodec_autoencoder_24k_320d")
        self.encoder_config_path = "audiodec_autoencoder_24k_320d/checkpoint-500000steps.pkl"

        # download decoder
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/checkpoint-500000steps.pkl',
            'audiodec_vocoder_24k_320d')
        nlp2.download_file(
            'https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/config.yml',
            "audiodec_vocoder_24k_320d")
        nlp2.download_file(
            "https://huggingface.co/AudioDecBenchmark/AudioDec/resolve/main/vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean/symAD_libritts_24000_hop300_clean.npy",
            "audiodec_vocoder_24k_320d"
        )
        self.decoder_config_path = "audiodec_vocoder_24k_320d/checkpoint-500000steps.pkl"
        self.sampling_rate = 24000
        audiodec = AudioDecModel(tx_device=self.device, rx_device=self.device)
        audiodec.load_transmitter(self.encoder_config_path)
        audiodec.load_receiver(self.encoder_config_path, self.decoder_config_path)
        self.model = audiodec

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        audio_array = self.decode_unit(extracted_unit.stuff_for_synth)
        data['unit'] = extracted_unit.unit
        if local_save:
            audio_path = f"dummy_{self.setting}/{data['id']}.wav"
            save_audio(audio_array, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_array
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        x = torch.from_numpy(data['audio']['array']).unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.device)
        self.model.tx_encoder.reset_buffer()
        z = self.model.tx_encoder.encode(x)
        zq, codes = self.model.tx_encoder.quantizer.codebook.forward_index(z.transpose(2, 1), flatten_idx=False)
        if len(codes.shape) == 2:
            codes = codes.unsqueeze(1)
        codes = codes.transpose(0, 1).squeeze()
        codebook_size = self.model.rx_encoder.quantizer.codebook.codebook_size
        for y, code in enumerate(codes):
            codes[y] += int(y * codebook_size)
        codes = codes.squeeze(0)
        return ExtractedUnit(
            unit=codes,
            stuff_for_synth=(zq, codes)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        _, codes = stuff_for_synth
        zq = self.model.rx_encoder.lookup(codes)
        y = self.model.decoder.decode(zq)
        return y[0].cpu().detach().numpy()
