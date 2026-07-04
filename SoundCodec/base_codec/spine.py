import numpy as np
import torch

from SoundCodec.base_codec.general import BaseCodec, ExtractedUnit


class SpineBaseCodec(BaseCodec):
    """Spine: a neural audio codec for expressive speech (24kHz, multi-scale FSQ)."""

    def __init__(self):
        super().__init__()
        try:
            from spine import Spine
        except ImportError:
            raise Exception("Please install spine first. pip install spine-codec")
        self.model = Spine.from_pretrained(self.repo_id, device=self.device)

    def config(self):
        self.repo_id = "twangodev/spine-codec"
        self.sampling_rate = 24_000

    @torch.no_grad()
    def extract_unit(self, data):
        wav = np.asarray(data["audio"]["array"], dtype=np.float32)
        x = torch.from_numpy(wav).to(self.device).view(1, 1, -1)
        # One token stream per temporal scale; flatten coarse-to-fine for the unit view.
        codes = self.model.encode(x)
        return ExtractedUnit(
            unit=torch.cat([c.flatten() for c in codes]).cpu(),
            stuff_for_synth=(codes, x.shape[-1]),
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        codes, length = stuff_for_synth
        wav = self.model.decode(codes)[..., :length]
        return wav.squeeze().detach().cpu().numpy()
