import torch
import numpy as np
import nlp2
from SoundCodec.base_codec.general import BaseCodec, ExtractedUnit, BatchExtractedUnit
from huggingface_hub import hf_hub_download
import os

class AUVBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()

    def config(self):
        self._setup_model()

    def _setup_model(self):
        try:
            from auv.model import AUV
        except ImportError:
            raise Exception("Please install auv first: pip install auv")

        self._download_resources()
        self.model = AUV()
        self.model.from_pretrained(self.ckpt_path, device=self.device)
        self.model.eval()
        self.model.to(self.device)
        self.sampling_rate = 16000 # Default SR for AUV

    def _download_resources(self):
        if hasattr(self, "ckpt_repo") and hasattr(self, "ckpt_filename"):
            self.ckpt_path = hf_hub_download(repo_id=self.ckpt_repo, filename=self.ckpt_filename)
        elif hasattr(self, "ckpt_url"):
            self.ckpt_path = nlp2.download_file(self.ckpt_url, "auv_model")
        else:
            # Default checkpoint
            self.ckpt_path = hf_hub_download(repo_id="SWivid/AUV", filename="auv.pt")

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(np.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        sr = data["audio"]["sampling_rate"]
        
        # AUV encode expects a dict with sample and sample_rate
        input_data = {
            "sample": wav,
            "sample_rate": sr
        }
        
        enc_res = self.model.encode(input_data)
        tokens = enc_res["tokens"] # [1, L, Q] or similar
        
        return ExtractedUnit(
            unit=tokens.squeeze(),
            stuff_for_synth=enc_res["quantized"]
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        quantized = stuff_for_synth
        recon = self.model.decode(quantized)
        return recon[0].cpu().numpy()

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        # AUV model seems to support only batch_size 1 in its encode method according to its source
        # So we iterate for now
        units = []
        stuff_for_synth = []
        for data in data_list:
            unit = self.extract_unit(data)
            units.append(unit.unit)
            stuff_for_synth.append(unit.stuff_for_synth)
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        results = []
        for stuff in batch_extracted_unit.stuff_for_synth:
            results.append(self.decode_unit(stuff))
        return results
