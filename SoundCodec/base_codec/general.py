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


def resample_audio(audio_array, orig_sr, target_sr):
    """
    Resample audio from orig_sr to target_sr.
    
    Args:
        audio_array: numpy array or torch tensor of shape (samples,) or (samples, channels)
        orig_sr: original sampling rate (int)
        target_sr: target sampling rate (int)
    
    Returns:
        resampled audio as numpy array
    """
    if orig_sr == target_sr:
        return audio_array
    
    # Convert to numpy if torch tensor
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.cpu().numpy()
    
    # Use librosa for resampling
    try:
        import librosa
        # librosa.resample expects (n_samples,) or (n_channels, n_samples)
        # Handle both shapes
        if audio_array.ndim == 1:
            # Simple 1D: (samples,)
            resampled = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
        elif audio_array.ndim == 2:
            # 2D: resample each row/channel independently
            # Assume shape is (batch/channels, samples)
            resampled = np.array([
                librosa.resample(row, orig_sr=orig_sr, target_sr=target_sr)
                for row in audio_array
            ])
        else:
            raise ValueError(f"Unexpected audio array shape: {audio_array.shape}")
        
        return resampled
    except ImportError:
        raise ImportError("librosa is required for resampling. Install with: pip install librosa")


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
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    # Handle potentially empty or zero-length tensors gracefully
    if wav.numel() == 0:
        print(f"Warning: Attempting to save empty audio to {path}. Skipping.")
        return

    
    # Move to CPU immediately to avoid GPU-related C++ errors in backends
    wav = wav.detach().cpu()
    
    # Check for NaN or Inf which can crash some backends
    if torch.isnan(wav).any() or torch.isinf(wav).any():
        print(f"Warning: Audio contains NaN or Inf. Clamping and replacing NaNs with zeros.")
        wav = torch.nan_to_num(wav, nan=0.0, posinf=limit, neginf=-limit)

    limit = 0.99
    try:
        max_val = wav.abs().max()
        if max_val > 0:
            wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
        
        # Try scipy first as it's pure Python/NumPy (no C++ backend issues)
        try:
            from scipy.io import wavfile
            audio_out = wav.squeeze().numpy()
            audio_int16 = (audio_out * 32767).astype(np.int16)
            wavfile.write(str(path), sample_rate, audio_int16)
            return
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"Scipy save failed: {e}. Falling back to torchaudio.")
        
        # Try soundfile as second option
        try:
            import soundfile as sf
            audio_out = wav.squeeze().numpy()
            sf.write(str(path), audio_out, sample_rate, subtype='PCM_16')
            return
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"Soundfile save failed: {e}. Falling back to torchaudio.")

        # Last resort: torchaudio
        torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
    except Exception as e:
        print(f"Error saving audio to {path}: {e}")
        raise


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
        # Default to True, subclasses can override to False for encode-only codecs
        if not hasattr(self, 'supports_decode'):
            self.supports_decode = True
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
        """Synthesize audio from data for a single sample with automatic resampling."""
        # Get original sampling rate
        orig_sr = data.get('audio', {}).get('sampling_rate')
        codec_sr = self.sampling_rate
        
        # Prepare data for codec processing
        if orig_sr and codec_sr and orig_sr != codec_sr:
            # Need to resample to codec's native sampling rate
            orig_audio = data['audio']['array']
            if isinstance(orig_audio, torch.Tensor):
                orig_audio = orig_audio.cpu().numpy()
            
            # Resample to codec SR
            resampled_audio = resample_audio(orig_audio, orig_sr, codec_sr)
            
            # Create temporary data with resampled audio
            temp_data = data.copy()
            temp_data['audio'] = {
                'array': resampled_audio,
                'sampling_rate': codec_sr
            }
        else:
            temp_data = data
            if not orig_sr:
                # If no original SR, assume codec SR
                orig_sr = codec_sr
        
        # Extract unit and decode at codec's native SR
        extracted_unit = self.extract_unit(temp_data)
        temp_data['unit'] = extracted_unit.unit
        
        # Check if codec supports decoding
        if not self.supports_decode:
            # For encode-only codecs, return data with unit but no audio reconstruction
            if local_save:
                pass
            else:
                data['audio']['array'] = None
                data['encode_only'] = True
            return data
        
        # Decode at codec SR
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        
        # Convert to numpy if needed
        if isinstance(audio_values, torch.Tensor):
            audio_values = audio_values.cpu().numpy()
        
        # Resample back to original SR if needed
        if orig_sr != codec_sr:
            audio_values = resample_audio(audio_values, codec_sr, orig_sr)
        
        # Save or return
        if local_save:
            audio_id = data.get('id', str(uuid.uuid4()))
            audio_path = f"dummy_{self.setting}/{audio_id}.wav"
            save_audio(audio_values, audio_path, orig_sr)  # Save at original SR
            data['audio'] = audio_path
        else:
            data['audio'] = {
                'array': audio_values,
                'sampling_rate': orig_sr  # CRITICAL: preserve original SR
            }
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
        
        # Check if codec supports decoding
        if not self.supports_decode:
            # For encode-only codecs, return data with units but no audio reconstruction
            for i, (data, unit) in enumerate(zip(data_list, batch_extracted_unit.units)):
                data['unit'] = unit
                if not local_save:
                    data['audio']['array'] = None
                    data['encode_only'] = True
            return data_list
        
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
