import os
from dataclasses import dataclass

import numpy as np
import torchaudio

import torch


def pad_arrays_to_match(array1, array2):
    shape1, shape2 = array1.shape, array2.shape
    if len(shape1) != len(shape2):
        raise ValueError("The two arrays must have the same number of dimensions")
    padding1 = [(0, max(dim_size2 - dim_size1, 0)) for dim_size1, dim_size2 in zip(shape1, shape2)]
    padding2 = [(0, max(dim_size1 - dim_size2, 0)) for dim_size1, dim_size2 in zip(shape1, shape2)]
    array1_padded = np.pad(array1, padding1, mode='constant')
    array2_padded = np.pad(array2, padding2, mode='constant')
    return array1_padded, array2_padded


@dataclass
class ExtractedUnit:
    unit: torch.Tensor
    stuff_for_synth: object

    def to_dict(self):
        return {
            'unit': self.unit,  # torch.Tensor with shape [codec_layer, time_dim]
            'stuff_for_synth': self.stuff_for_synth
        }


def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
