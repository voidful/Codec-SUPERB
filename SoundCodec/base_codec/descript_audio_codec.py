from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit
import torch
from audiotools import AudioSignal
import numpy as np


class DACBaseCodec(BaseCodec):
    def __init__(self):
        # Reference: https://github.com/descriptinc/descript-audio-codec
        super().__init__()
        import dac
        self.model_path = dac.utils.download(model_type=self.model_type)
        self.model = dac.DAC.load(self.model_path)
        self.model.to(self.device)

    def config(self):
        self.model_type = "24khz"
        self.sampling_rate = 24_000
        try:
            import dac
        except:
            raise Exception("Please install descript-audio-codec first. pip install descript-audio-codec")

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        compressed_audio, unit_only = extracted_unit.stuff_for_synth
        data['unit'] = extracted_unit.unit
        decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
        if local_save:
            from SoundCodec.base_codec.general import uuid
            audio_id = data.get('id', str(uuid.uuid4()))
            audio_path = f"dummy-descript-audio-codec-{self.model_type}/{audio_id}.wav"
            save_audio(decompressed_audio, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = decompressed_audio.cpu().numpy()
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        audio_array = data["audio"]["array"]
        if len(audio_array.shape) == 1:
            audio_array = audio_array[None, :]
        input_signal = AudioSignal(audio_array, self.sampling_rate)
        input_signal.to(self.device)
        
        # Encode audio signal as a compressed stream
        compressed_audio = self.model.compress(input_signal)
        # Create discretized representation
        codes = compressed_audio.codes.squeeze(0).permute(1, 0)
        return ExtractedUnit(
            unit=codes,
            stuff_for_synth=(compressed_audio, True)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        compressed_audio, unit_only = stuff_for_synth
        decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
        return decompressed_audio.cpu().numpy()

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Batch extraction for Descript Audio Codec."""
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
            wav = torch.tensor(data["audio"]["array"]).to(torch.float32)
            wav_list.append(wav)
        
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
        
        # Compress the entire batch at once
        with torch.no_grad():
            batch_compressed_audio = self.model.compress(batch_wav)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract compressed audio for this item
            item_compressed = []
            for compressed_item in batch_compressed_audio:
                # Extract the i-th item from each compressed component
                if hasattr(compressed_item, 'codes'):
                    item_codes = compressed_item.codes[i:i+1]
                    item_compressed_part = type(compressed_item)(codes=item_codes)
                    if hasattr(compressed_item, 'length'):
                        item_compressed_part.length = compressed_item.length[i:i+1] if compressed_item.length.dim() > 0 else compressed_item.length
                    item_compressed.append(item_compressed_part)
                else:
                    item_compressed.append(compressed_item[i:i+1])
            
            # Create codes representation
            codes = batch_compressed_audio[0].codes[i].squeeze(0).permute(1, 0)
            
            units.append(codes)
            stuff_for_synth.append((item_compressed, True))
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Batch decoding for Descript Audio Codec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        audio_values = []
        
        # Process each item individually due to complex compressed audio structure
        # Could potentially be optimized further by reconstructing batch compressed audio
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            compressed_audio, unit_only = stuff_for_synth
            decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)
            audio_values.append(decompressed_audio.cpu().numpy())
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = DACBaseCodec
