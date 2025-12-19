import os
from dataclasses import dataclass
from typing import List, Union, Any
from abc import ABC, abstractmethod
import uuid

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
            'unit': self.unit,  # torch.Tensor with shape [codec_layer, time_dim] or [batch, codec_layer, time_dim]
            'stuff_for_synth': self.stuff_for_synth
        }


@dataclass
class BatchExtractedUnit:
    units: List[torch.Tensor]  # List of units, one per batch item
    stuff_for_synth: List[Any]  # List of synthesis data, one per batch item
    batch_size: int

    def to_dict(self):
        return {
            'units': self.units,
            'stuff_for_synth': self.stuff_for_synth,
            'batch_size': self.batch_size
        }


def save_audio(wav: Union[torch.Tensor, np.ndarray], path, sample_rate: int, rescale: bool = False):
    if sample_rate is None:
        raise ValueError(f"sample_rate cannot be None when saving audio to {path}")
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(f"Saving audio to {path} with sample_rate {sample_rate}")
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


class BaseCodec(ABC):
    """Base class for all audio codecs with batch support."""
    
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.sampling_rate = None
        self.setting = None
        self.config()
    
    @abstractmethod
    def config(self):
        """Configure the codec (model loading, settings, etc.)"""
        pass
    
    @abstractmethod
    def extract_unit(self, data):
        """Extract units from a single audio sample."""
        pass
    
    @abstractmethod
    def decode_unit(self, stuff_for_synth):
        """Decode units back to audio for a single sample."""
        pass
    
    def is_1d(self):
        """Identify whether the codec produces 1D tokens using a dummy extraction."""
        dummy_data = {
            "audio": {
                "array": np.zeros(16000),
                "sampling_rate": 16000
            }
        }
        try:
            extracted_unit = self.extract_unit(dummy_data)
            unit = extracted_unit.unit
            if hasattr(unit, 'squeeze'):
                unit = unit.squeeze()
            
            # Check for effective ndim (dimensions with size > 1)
            effective_ndim = len([d for d in unit.shape if d > 1])
            return effective_ndim <= 1
        except Exception:
            return False

    def synth(self, data, local_save=True):
        """Synthesize audio from data for a single sample."""
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_id = data.get('id', str(uuid.uuid4()))
            audio_path = f"dummy_{self.setting}/{audio_id}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data
    
    def batch_extract_unit(self, data_list: List[dict]) -> BatchExtractedUnit:
        """
        Extract units from a batch of audio samples.
        Default implementation uses a loop - override for better performance.
        """
        units = []
        stuff_for_synth = []
        
        for data in data_list:
            extracted_unit = self.extract_unit(data)
            units.append(extracted_unit.unit)
            stuff_for_synth.append(extracted_unit.stuff_for_synth)
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )
    
    def batch_decode_unit(self, batch_extracted_unit: BatchExtractedUnit) -> List[np.ndarray]:
        """
        Decode a batch of units back to audio.
        Default implementation uses a loop - override for better performance.
        """
        audio_values = []
        
        for stuff_for_synth in batch_extracted_unit.stuff_for_synth:
            audio = self.decode_unit(stuff_for_synth)
            audio_values.append(audio)
        
        return audio_values
    
    def batch_synth(self, data_list: List[dict], local_save=True) -> List[dict]:
        """
        Synthesize audio from a batch of data.
        Default implementation uses batch_extract_unit and batch_decode_unit.
        """
        batch_extracted_unit = self.batch_extract_unit(data_list)
        batch_audio_values = self.batch_decode_unit(batch_extracted_unit)
        
        # Update the data list with results
        for i, (data, audio_values, unit) in enumerate(zip(data_list, batch_audio_values, batch_extracted_unit.units)):
            data['unit'] = unit
            if local_save:
                audio_id = data.get('id', str(uuid.uuid4()))
                audio_path = f"dummy_{self.setting}/{audio_id}.wav"
                save_audio(torch.tensor(audio_values), audio_path, self.sampling_rate)
                data['audio'] = audio_path
            else:
                data['audio']['array'] = audio_values
        
        return data_list
