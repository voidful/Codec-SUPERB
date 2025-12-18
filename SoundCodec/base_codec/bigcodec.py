import torch
import numpy as np
import nlp2
from SoundCodec.base_codec.general import BaseCodec, ExtractedUnit, BatchExtractedUnit
from huggingface_hub import hf_hub_download
import os

class BigCodecBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()

    def config(self):
        self._setup_model()

    def _setup_model(self):
        try:
            from bigcodec import CodecEncoder, CodecDecoder
        except ImportError:
            raise Exception("Please install bigcodec first: pip install bigcodec")

        self._download_resources()
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        
        self.encoder = CodecEncoder()
        self.encoder.load_state_dict(ckpt['CodecEnc'])
        self.encoder.eval().to(self.device)
        
        self.decoder = CodecDecoder()
        self.decoder.load_state_dict(ckpt['generator'])
        self.decoder.eval().to(self.device)
        
        self.sampling_rate = 16000 # Default SR for BigCodec

    def _download_resources(self):
        if hasattr(self, "ckpt_repo") and hasattr(self, "ckpt_filename"):
            self.ckpt_path = hf_hub_download(repo_id=self.ckpt_repo, filename=self.ckpt_filename)
        elif hasattr(self, "ckpt_url"):
            self.ckpt_path = nlp2.download_file(self.ckpt_url, "bigcodec_model")
        else:
            # Default checkpoint
            self.ckpt_path = hf_hub_download(repo_id="Alethia/BigCodec", filename="bigcodec.pt")

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(np.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        # BigCodec expects [B, 1, T]
        wav = wav.unsqueeze(1)
        
        # Pad to multiple of 200 as in official inference script
        hop = 200
        pad = hop - (wav.shape[2] % hop)
        if pad != hop:
            wav = torch.nn.functional.pad(wav, (0, pad))
            
        vq_emb = self.encoder(wav)
        vq_post_emb, vq_code, _ = self.decoder(vq_emb, vq=True)
        
        return ExtractedUnit(
            unit=vq_code.squeeze(0),
            stuff_for_synth=vq_post_emb
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        vq_post_emb = stuff_for_synth
        recon = self.decoder(vq_post_emb, vq=False)
        return recon.squeeze().cpu().numpy()

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
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
