# Reference: https://github.com/zhai-lw/SQCodec

from functools import lru_cache

import torch
import numpy as np
import torchaudio.transforms as T
from SoundCodec.base_codec.general import ExtractedUnit, save_audio, BaseCodec, BatchExtractedUnit
import nlp2


class SQCodecBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()

    def config(self):
        import sqcodec
        self.config_name = "1k5bps"
        self.model = sqcodec.SQCodec.load_model(device=self.device, config_name=self.config_name)
        self.sampling_rate = self.model.sample_rate
        self.resample_func = lambda target_sr: T.Resample(orig_freq=target_sr, new_freq=self.sampling_rate)

    def extract_unit(self, data_item):
        wav, sr = data_item["audio"]["array"], data_item["audio"]["sampling_rate"]
        wav = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            audio_data = self.resample_func(sr)(wav)
            audio_data, audio_length = self.model.preprocess(audio_data)
            feature = self.model.encoder(audio_data.unsqueeze(1))
            trans_feature = self.model.en_encoder(feature)
            q_trans_feature, indices, _ = self.model.quantizer(trans_feature)

        return ExtractedUnit(
            unit=indices[0].unsqueeze(0),
            stuff_for_synth=(q_trans_feature, audio_length),
        )

    def decode_unit(self, stuff_for_synth):
        q_trans_feature, audio_length = stuff_for_synth

        with torch.inference_mode():
            q_feature = self.model.en_decoder(q_trans_feature)
            audio_data = self.model.decoder(q_feature).squeeze(1)

        return audio_data[:, :audio_length].cpu().numpy()

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Batch extraction for SQCodec."""
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
        original_lengths = []
        
        for data_item in data_list:
            wav, sr = data_item["audio"]["array"], data_item["audio"]["sampling_rate"]
            wav = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)
            # Resample if needed
            audio_data = self.resample_func(sr)(wav)
            wav_list.append(audio_data)
            original_lengths.append(audio_data.shape[-1])
        
        # Pad all waveforms to the same length
        max_length = max(wav.shape[-1] for wav in wav_list)
        padded_wavs = []
        audio_lengths = []
        
        for wav in wav_list:
            if wav.shape[-1] < max_length:
                padding = max_length - wav.shape[-1]
                wav = torch.nn.functional.pad(wav, (0, padding))
            
            # Preprocess individual items
            processed_wav, audio_length = self.model.preprocess(wav)
            padded_wavs.append(processed_wav)
            audio_lengths.append(audio_length)
        
        # Stack into batch tensor [B, T]
        batch_wav = torch.stack(padded_wavs, dim=0)
        
        # Encode the entire batch
        with torch.inference_mode():
            feature = self.model.encoder(batch_wav.unsqueeze(2))  # Add channel dimension
            trans_feature = self.model.en_encoder(feature)
            q_trans_feature, indices, _ = self.model.quantizer(trans_feature)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract features for this item
            item_q_trans_feature = q_trans_feature[i:i+1]
            item_indices = indices[i:i+1]
            
            units.append(item_indices[0].unsqueeze(0))
            stuff_for_synth.append((item_q_trans_feature, audio_lengths[i]))
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Batch decoding for SQCodec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        audio_values = []
        
        # Process each item individually due to variable audio lengths
        # Could potentially be optimized further by grouping items with similar lengths
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            q_trans_feature, audio_length = stuff_for_synth
            
            with torch.inference_mode():
                q_feature = self.model.en_decoder(q_trans_feature)
                audio_data = self.model.decoder(q_feature).squeeze(1)
            
            audio_values.append(audio_data[:, :audio_length].cpu().numpy())
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = SQCodecBaseCodec
