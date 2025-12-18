import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit


class EncodecHFBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(self.pretrained_model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name)
        self.sampling_rate = self.processor.sampling_rate

    def config(self):
        self.pretrained_model_name = "facebook/encodec_24khz"

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            from SoundCodec.base_codec.general import uuid
            audio_id = data.get('id', str(uuid.uuid4()))
            audio_path = f"dummy_{self.pretrained_model_name.replace('/', '_')}/{audio_id}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        audio_sample = data["audio"]["array"]
        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device) if inputs["padding_mask"] is not None else None
        encoder_outputs = self.model.encode(input_values, padding_mask)
        return ExtractedUnit(
            unit=encoder_outputs.audio_codes.squeeze(),
            stuff_for_synth=(encoder_outputs, padding_mask)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        encoder_outputs, padding_mask = stuff_for_synth
        audio_values = \
            self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, padding_mask)[0]
        return audio_values[0].cpu().numpy()

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Optimized batch extraction for HuggingFace EnCodec."""
        if len(data_list) == 1:
            # Single item, use regular method
            extracted_unit = self.extract_unit(data_list[0])
            return BatchExtractedUnit(
                units=[extracted_unit.unit],
                stuff_for_synth=[extracted_unit.stuff_for_synth],
                batch_size=1
            )
        
        # Prepare batch data
        audio_samples = [data["audio"]["array"] for data in data_list]
        
        # Process all samples with the processor (it handles batching)
        inputs = self.processor(
            raw_audio=audio_samples, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt",
            padding=True
        )
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device) if inputs["padding_mask"] is not None else None
        
        # Encode the entire batch at once
        with torch.no_grad():
            batch_encoder_outputs = self.model.encode(input_values, padding_mask)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract outputs for this item
            item_audio_codes = batch_encoder_outputs.audio_codes[i:i+1]
            item_audio_scales = batch_encoder_outputs.audio_scales[i:i+1] if hasattr(batch_encoder_outputs, 'audio_scales') else None
            item_padding_mask = padding_mask[i:i+1] if padding_mask is not None else None
            
            # Create a single-item encoder output structure
            from types import SimpleNamespace
            item_encoder_outputs = SimpleNamespace(
                audio_codes=item_audio_codes,
                audio_scales=item_audio_scales
            )
            
            units.append(item_audio_codes.squeeze())
            stuff_for_synth.append((item_encoder_outputs, item_padding_mask))
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Optimized batch decoding for HuggingFace EnCodec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        # Collect all encoder outputs and padding masks for batch processing
        all_audio_codes = []
        all_audio_scales = []
        all_padding_masks = []
        
        for encoder_outputs, padding_mask in batch_extracted_unit.stuff_for_synth:
            all_audio_codes.append(encoder_outputs.audio_codes)
            if hasattr(encoder_outputs, 'audio_scales') and encoder_outputs.audio_scales is not None:
                all_audio_scales.append(encoder_outputs.audio_scales)
            if padding_mask is not None:
                all_padding_masks.append(padding_mask)
        
        # Stack for batch processing
        batch_audio_codes = torch.cat(all_audio_codes, dim=0)
        batch_audio_scales = torch.cat(all_audio_scales, dim=0) if all_audio_scales else None
        batch_padding_mask = torch.cat(all_padding_masks, dim=0) if all_padding_masks else None
        
        # Decode the entire batch at once
        with torch.no_grad():
            batch_audio_values = self.model.decode(batch_audio_codes, batch_audio_scales, batch_padding_mask)[0]
        
        # Split results for each item in the batch
        audio_values = []
        for i in range(batch_extracted_unit.batch_size):
            audio_values.append(batch_audio_values[i].cpu().numpy())
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = EncodecHFBaseCodec
