import numpy as np
import soundfile as sf
import mir_eval
import librosa
import glob
import os

def SDR_cal(ref_folder, est_folder, target_sr=None):
    """
    Calculate the Signal-to-Distortion Ratio (SDR) for pairs of reference and estimated audio files
    located in the given directories, optionally resampling all files to a specified sample rate and aligning
    them using Dynamic Time Warping (DTW).
    
    Parameters:
        ref_folder (str): The folder path containing the reference audio files (.wav).
        est_folder (str): The folder path containing the estimated/generated audio files (.wav).
        target_sr (int, optional): The target sample rate to which all audio files will be resampled. If None, no resampling is performed.
    
    Returns:
        dict: A dictionary containing the SDR values for each pair of audio files, with file names as keys.
    """
    ref_files = sorted(glob.glob(os.path.join(ref_folder, '*.wav')))
    est_files = sorted(glob.glob(os.path.join(est_folder, '*.wav')))
    
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    
    sdr_scores = {}
    mean_score = []
    for ref_path, est_path in zip(ref_files, est_files):
        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)

        if target_sr is not None:
            if ref_rate != target_sr:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_rate, target_sr=target_sr)
            if est_rate != target_sr:
                est_audio = librosa.resample(est_audio, orig_sr=est_rate, target_sr=target_sr)

        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]
        if est_audio.ndim > 1:
            est_audio = est_audio[:, 0]

        min_len = min(len(ref_audio), len(est_audio))
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]
        
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref_audio[None, :], est_audio[None, :], compute_permutation=False)
        sdr_scores[os.path.basename(ref_path)] = sdr[0]
        mean_score.append(sdr[0])
    
    return sdr_scores, np.mean(mean_score)
