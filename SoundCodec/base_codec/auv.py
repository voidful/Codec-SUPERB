import torch
import numpy as np
import nlp2
from SoundCodec.base_codec.general import BaseCodec, ExtractedUnit, BatchExtractedUnit
from huggingface_hub import hf_hub_download


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
        self.sampling_rate = 16000  # Default SR for AUV

    def _download_resources(self):
        if hasattr(self, "ckpt_repo") and hasattr(self, "ckpt_filename"):
            self.ckpt_path = hf_hub_download(repo_id=self.ckpt_repo, filename=self.ckpt_filename)
        elif hasattr(self, "ckpt_url"):
            self.ckpt_path = nlp2.download_file(self.ckpt_url, "auv_model")
        else:
            self.ckpt_path = hf_hub_download(repo_id="SWivid/AUV", filename="auv.pt")

    @staticmethod
    def _squeeze_tokens(tokens):
        # AUV's num_quantizers default = 1; collapse the trailing quantizer dim so
        # is_1d() and downstream metrics treat the unit as a 1D token stream.
        if tokens.ndim > 1 and tokens.shape[-1] == 1:
            tokens = tokens.squeeze(-1)
        return tokens

    @torch.no_grad()
    def extract_unit(self, data):
        wav_arr = np.asarray(data["audio"]["array"])
        raw_length = int(wav_arr.shape[0])
        wav = torch.as_tensor(wav_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        sr = data["audio"]["sampling_rate"]

        enc_res = self.model.encode({"sample": wav, "sample_rate": sr})

        tokens = self._squeeze_tokens(enc_res["tokens"].squeeze(0))

        return ExtractedUnit(
            unit=tokens,
            stuff_for_synth=(enc_res["quantized"], raw_length),
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        quantized, raw_length = stuff_for_synth
        recon = self.model.decode(quantized)
        audio = recon[0].cpu().numpy()
        # Trim padding tail introduced by hop_length ceiling in the encoder.
        audio = audio[..., :raw_length]
        return audio

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        if len(data_list) == 0:
            return BatchExtractedUnit(units=[], stuff_for_synth=[], batch_size=0)

        if len(data_list) == 1:
            extracted = self.extract_unit(data_list[0])
            return BatchExtractedUnit(
                units=[extracted.unit],
                stuff_for_synth=[extracted.stuff_for_synth],
                batch_size=1,
            )

        sr = data_list[0]["audio"]["sampling_rate"]
        wavs = []
        raw_lengths = []
        for data in data_list:
            if data["audio"]["sampling_rate"] != sr:
                raise ValueError(
                    "batch_extract_unit requires all samples to share the same sampling_rate; "
                    f"got {sr} and {data['audio']['sampling_rate']}"
                )
            wav = torch.as_tensor(np.asarray(data["audio"]["array"]), dtype=torch.float32)
            wavs.append(wav)
            raw_lengths.append(int(wav.shape[0]))

        max_len = max(raw_lengths)
        padded = torch.stack(
            [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in wavs],
            dim=0,
        ).to(self.device)
        lengths = torch.tensor(raw_lengths, device=self.device, dtype=torch.long)

        enc_res = self.model.encode({
            "sample": padded,
            "sample_rate": sr,
            "lengths": lengths,
        })

        tokens = enc_res["tokens"]          # [B, L, Q] (Q=1 for default config)
        quantized = enc_res["quantized"]    # [B, C, L]
        feat_lengths = enc_res["lengths"]   # [B]

        units = []
        stuff_for_synth = []
        for i in range(len(data_list)):
            feat_len = int(feat_lengths[i].item())
            units.append(self._squeeze_tokens(tokens[i, :feat_len]))
            q_i = quantized[i:i + 1, :, :feat_len].contiguous()
            stuff_for_synth.append((q_i, raw_lengths[i]))

        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list),
        )
