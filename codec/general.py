import torch
import torchaudio


def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
