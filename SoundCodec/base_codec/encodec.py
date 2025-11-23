import torch
import numpy as np
from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit


class EncodecBaseCodec(BaseCodec):
    def __init__(self):
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            self.EncodecModel = EncodecModel
            self.convert_audio = convert_audio
        except ImportError:
            raise Exception("Please install encodec first. pip install encodec")
        self.model = None
        super().__init__()

    def config(self):
        if self.model is None:
            self.model = self.EncodecModel.encodec_model_24khz().to(self.device)
        self.model.set_target_bandwidth(6.0)
        self.setting = "encodec_24khz_6.0"
        self.sampling_rate = 24_000

    @torch.no_grad()
    def extract_unit(self, data):
        wav, sr = data["audio"]["array"], data["audio"]["sampling_rate"]
        # unsqueeze to [B, T], if no batch, B=1
        wav = torch.tensor(wav).unsqueeze(0)
        wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32).to(self.device)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [B, n_q, T]
        return ExtractedUnit(
            unit=codes,
            stuff_for_synth=(encoded_frames, data['audio']['array'].shape[0])
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        encoded_frames, original_shape = stuff_for_synth
        audio_values = self.model.decode(encoded_frames)[0]
        # trim the audio to the same length as the input
        audio_values = audio_values[:, :original_shape].cpu().numpy()
        return audio_values

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Optimized batch extraction for EnCodec."""
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
        original_shapes = []
        
        for data in data_list:
            wav, sr = data["audio"]["array"], data["audio"]["sampling_rate"]
            wav = torch.tensor(wav).to(torch.float32)
            wav_list.append(wav)
            original_shapes.append(wav.shape[0])
        
        # Pad all waveforms to the same length
        max_length = max(wav.shape[0] for wav in wav_list)
        padded_wavs = []
        for wav in wav_list:
            if wav.shape[0] < max_length:
                padding = max_length - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, padding))
            padded_wavs.append(wav.unsqueeze(0))  # Add channel dimension
        
        # Stack into batch tensor [B, 1, T]
        batch_wav = torch.stack(padded_wavs, dim=0).to(self.device)
        
        # Encode the entire batch at once
        with torch.no_grad():
            batch_encoded_frames = self.model.encode(batch_wav)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract codes for this item
            item_encoded_frames = [(frame[i:i+1] for frame in encoded_frame) for encoded_frame in batch_encoded_frames]
            codes = torch.cat([encoded[0] for encoded in item_encoded_frames], dim=-1).squeeze()
            units.append(codes)
            stuff_for_synth.append((item_encoded_frames, original_shapes[i]))
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Optimized batch decoding for EnCodec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        audio_values = []
        
        # Group items by similar encoded frame structures for potential batching
        # For now, process individually but could be optimized further
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            encoded_frames, original_shape = stuff_for_synth
            audio = self.model.decode(encoded_frames)[0]
            audio = audio[:, :original_shape].cpu().numpy()
            audio_values.append(audio)
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = EncodecBaseCodec
