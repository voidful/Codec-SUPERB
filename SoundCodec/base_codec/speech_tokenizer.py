import numpy
import torch

from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit
import torchaudio
import nlp2


class SpeechTokenizerBaseCodec(BaseCodec):
    def __init__(self):
        try:
            from speechtokenizer import SpeechTokenizer
        except Exception as e:
            raise Exception(f"Please install SpeechTokenizer first. pip install -U speechtokenizer. Error: {e}")

        super().__init__()
        self.model = SpeechTokenizer.load_from_checkpoint(self.config_path, self.ckpt_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.sampling_rate = self.model.sample_rate

    def config(self):
        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/raw/main/speechtokenizer_hubert_avg/config.json',
            'speechtokenizer_hubert_avg')
        self.config_path = "speechtokenizer_hubert_avg/config.json"

        nlp2.download_file(
            'https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt',
            "speechtokenizer_hubert_avg")
        self.ckpt_path = "speechtokenizer_hubert_avg/SpeechTokenizer.pt"

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(numpy.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        wav = wav.unsqueeze(0)
        codes = self.model.encode(wav.to(self.device))
        return ExtractedUnit(
            unit=codes.permute(1, 0, 2).squeeze(0),
            stuff_for_synth=codes
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        codes = stuff_for_synth
        RVQ_1 = codes[:1, :, :]  # Contain content info, can be considered as semantic tokens
        RVQ_supplement = codes[1:, :, :]  # Contain timbre info, complete info lost by the first quantizer
        # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
        wav = self.model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0).to(self.device))
        wav = wav.detach().cpu().squeeze(0).numpy()
        return wav

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Optimized batch extraction for SpeechTokenizer."""
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
            wav = torch.tensor(numpy.array(data["audio"]['array']), dtype=torch.float32)
            wav_list.append(wav)
        
        # Pad all waveforms to the same length
        max_length = max(wav.shape[0] for wav in wav_list)
        padded_wavs = []
        for wav in wav_list:
            if wav.shape[0] < max_length:
                padding = max_length - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, padding))
            padded_wavs.append(wav)
        
        # Stack into batch tensor [B, T]
        batch_wav = torch.stack(padded_wavs, dim=0).to(self.device)
        
        # Encode the entire batch at once
        with torch.no_grad():
            batch_codes = self.model.encode(batch_wav)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract codes for this item
            item_codes = batch_codes[:, i:i+1, :]
            units.append(item_codes.permute(1, 0, 2).squeeze(0))
            stuff_for_synth.append(item_codes)
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Optimized batch decoding for SpeechTokenizer."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        audio_values = []
        
        # Process each item individually for now
        # Could be optimized further by batching the decode operation
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            codes = stuff_for_synth
            RVQ_1 = codes[:1, :, :]  # Contain content info
            RVQ_supplement = codes[1:, :, :]  # Contain timbre info
            # Concatenating and decoding
            wav = self.model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0).to(self.device))
            wav = wav.detach().cpu().squeeze(0).numpy()
            audio_values.append(wav)
        
        return audio_values


# For backward compatibility, keep the old class name
BaseCodec = SpeechTokenizerBaseCodec
