import nlp2
import torch
import os
import numpy as np

from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit
from audiotools import AudioSignal


class FunCodecBaseCodec(BaseCodec):
    def __init__(self):
        # Reference: https://github.com/alibaba-damo-academy/FunCodec
        try:
            from funcodec.bin.codec_inference import Speech2Token
        except:
            raise Exception(
                "Please install funcodec first. pip install git+https://github.com/voidful/FunCodec.git")
        os.makedirs("funcodec", exist_ok=True)
        super().__init__()
        self.config()

    def config(self):
        self.model_name = getattr(self, "model_name", "alibaba-damo/speech_codec-funcodec_en_libritts_16k_nq32ds320-pytorch")
        self.sampling_rate = getattr(self, "sampling_rate", 16000)
        
        if hasattr(self, 'config_path') and hasattr(self, 'ckpt_path'):
            from funcodec.bin.codec_inference import Speech2Token
            self.model = Speech2Token(
                config_file=self.config_path,
                model_file=self.ckpt_path,
                device=self.device,
            )
        else:
            try:
                from funasr import AutoModel
            except:
                raise Exception("Please install funasr first. pip install funasr")
            self.model = AutoModel(
                model=self.model_name,
                trust_remote_code=True,
                device=self.device,
            )

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_array = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy-funcodec-{self.model_name}/{data['id']}.wav"
            save_audio(audio_array, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_array
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        audio_signal = AudioSignal(data["audio"]['array'], data["audio"]['sampling_rate'])
        code_indices, code_embeddings, recon_speech, sub_quants = self.model(
            audio_signal.audio_data[0].to(self.device))
        return ExtractedUnit(
            unit=code_indices[0].permute(1, 0, 2).squeeze(0),
            stuff_for_synth={"code_indices": code_indices, "code_embeddings": code_embeddings,
                             "recon_speech": recon_speech}
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        extract_data = stuff_for_synth
        audio_array = extract_data["recon_speech"][0].cpu().numpy()
        return audio_array

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Batch extraction for FunCodec."""
        if len(data_list) == 1:
            # Single item, use regular method
            extracted_unit = self.extract_unit(data_list[0])
            return BatchExtractedUnit(
                units=[extracted_unit.unit],
                stuff_for_synth=[extracted_unit.stuff_for_synth],
                batch_size=1
            )
        
        # Prepare batch data
        audio_signals = []
        
        for data in data_list:
            audio_signal = AudioSignal(data["audio"]['array'], data["audio"]['sampling_rate'])
            audio_signals.append(audio_signal.audio_data[0])
        
        # Pad all audio signals to the same length
        max_length = max(audio.shape[-1] for audio in audio_signals)
        padded_audios = []
        for audio in audio_signals:
            if audio.shape[-1] < max_length:
                padding = max_length - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            padded_audios.append(audio)
        
        # Stack into batch tensor [B, C, T]
        batch_audio = torch.stack(padded_audios, dim=0).to(self.device)
        
        # Process the entire batch at once
        with torch.no_grad():
            batch_code_indices, batch_code_embeddings, batch_recon_speech, batch_sub_quants = self.model(batch_audio)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract results for this item
            item_code_indices = [code_idx[i:i+1] for code_idx in batch_code_indices]
            item_code_embeddings = [code_emb[i:i+1] for code_emb in batch_code_embeddings] if batch_code_embeddings else None
            item_recon_speech = [recon[i:i+1] for recon in batch_recon_speech]
            item_sub_quants = [sub_q[i:i+1] for sub_q in batch_sub_quants] if batch_sub_quants else None
            
            units.append(item_code_indices[0].permute(1, 0, 2).squeeze(0))
            stuff_for_synth.append({
                "code_indices": item_code_indices,
                "code_embeddings": item_code_embeddings,
                "recon_speech": item_recon_speech,
                "sub_quants": item_sub_quants
            })
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Batch decoding for FunCodec."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        audio_values = []
        
        # Process each item individually since recon_speech is already computed
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            extract_data = stuff_for_synth
            audio_array = extract_data["recon_speech"][0].cpu().numpy()
            audio_values.append(audio_array)
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = FunCodecBaseCodec
