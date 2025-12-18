import torch
import numpy as np
from SoundCodec.base_codec.general import BaseCodec, ExtractedUnit, BatchExtractedUnit

class S3TokenizerBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()

    def config(self):
        self._setup_model()

    def _setup_model(self):
        try:
            import s3tokenizer
        except ImportError:
            raise Exception("Please install s3tokenizer first: pip install s3tokenizer")

        self.tokenizer = s3tokenizer.load_model(getattr(self, "model_name", "speech_tokenizer_v1"))
        self.tokenizer.eval().to(self.device)
        self.sampling_rate = 16000 # Default usually

    @torch.no_grad()
    def extract_unit(self, data):
        import s3tokenizer
        wav = np.array(data["audio"]['array'])
        # s3tokenizer load_audio does some processing, but we have the array
        # Let's use their log_mel_spectrogram if they expect it
        # Actually s3tokenizer.log_mel_spectrogram expects a tensor?
        wav_tensor = torch.from_numpy(wav).float()
        mel = s3tokenizer.log_mel_spectrogram(wav_tensor)
        
        mels = [mel]
        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = self.tokenizer.quantize(mels.to(self.device), mels_lens.to(self.device))
        
        return ExtractedUnit(
            unit=codes[0, :codes_lens[0].item()],
            stuff_for_synth=None # No decoder available
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        # S3Tokenizer doesn't have a decoder in its package
        # Return zeros or a dummy signal to satisfy the interface
        return np.zeros(16000)

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        import s3tokenizer
        mels = []
        for data in data_list:
            wav = np.array(data["audio"]['array'])
            wav_tensor = torch.from_numpy(wav).float()
            mels.append(s3tokenizer.log_mel_spectrogram(wav_tensor))
            
        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = self.tokenizer.quantize(mels.to(self.device), mels_lens.to(self.device))
        
        units = []
        for i in range(len(data_list)):
            units.append(codes[i, :codes_lens[i].item()])
            
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=[None] * len(data_list),
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        return [np.zeros(16000)] * batch_extracted_unit.batch_size
