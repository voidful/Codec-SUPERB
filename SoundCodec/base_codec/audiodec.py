import nlp2
import torch
import numpy as np
from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit


class AudioDecBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()

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
        AudioDecModel = AudioDecModel
        audiodec = AudioDecModel(tx_device=self.device, rx_device=self.device)
        audiodec.load_transmitter(self.encoder_config_path)
        audiodec.load_receiver(self.encoder_config_path, self.decoder_config_path)
        self.model = audiodec

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

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Batch extraction for AudioDec."""
        if len(data_list) == 1:
            # Single item, use regular method
            extracted_unit = self.extract_unit(data_list[0])
            return BatchExtractedUnit(
                units=[extracted_unit.unit],
                stuff_for_synth=[extracted_unit.stuff_for_synth],
                batch_size=1
            )
        
        # Prepare batch data
        wav_list = []
        
        for data in data_list:
            wav = torch.from_numpy(data['audio']['array']).to(torch.float32)
            wav_list.append(wav)
        
        # Pad all waveforms to the same length
        max_length = max(wav.shape[0] for wav in wav_list)
        padded_wavs = []
        for wav in wav_list:
            if wav.shape[0] < max_length:
                padding = max_length - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, padding))
            padded_wavs.append(wav.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
        
        # Stack into batch tensor [B, 1, T]
        batch_wav = torch.cat(padded_wavs, dim=0).to(self.device)
        
        # Reset encoder buffer for batch processing
        self.model.tx_encoder.reset_buffer()
        
        # Encode the entire batch at once
        with torch.no_grad():
            z = self.model.tx_encoder.encode(batch_wav)
            zq, codes = self.model.tx_encoder.quantizer.codebook.forward_index(z.transpose(2, 1), flatten_idx=False)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract codes for this item
            item_codes = codes[i:i+1]
            if len(item_codes.shape) == 2:
                item_codes = item_codes.unsqueeze(1)
            item_codes = item_codes.transpose(0, 1).squeeze()
            
            # Apply codebook offset
            codebook_size = self.model.rx_encoder.quantizer.codebook.codebook_size
            for y, code in enumerate(item_codes):
                item_codes[y] += int(y * codebook_size)
            item_codes = item_codes.squeeze(0)
            
            # Extract zq for this item
            item_zq = zq[i:i+1]
            
            units.append(item_codes)
            stuff_for_synth.append((item_zq, item_codes))
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Batch decoding for AudioDec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        audio_values = []
        
        # Process each item individually due to the complex quantizer lookup
        # Could potentially be optimized further by batching the lookup and decode operations
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            _, codes = stuff_for_synth
            zq = self.model.rx_encoder.lookup(codes)
            y = self.model.decoder.decode(zq)
            audio_values.append(y[0].cpu().detach().numpy())
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = AudioDecBaseCodec
