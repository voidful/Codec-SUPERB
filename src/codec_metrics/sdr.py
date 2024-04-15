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
    # 获取所有参考音频和生成音频的路径
    ref_files = sorted(glob.glob(os.path.join(ref_folder, '*.wav')))
    est_files = sorted(glob.glob(os.path.join(est_folder, '*.wav')))
    
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    
    sdr_scores = {}
    mean_score = []
    for ref_path, est_path in zip(ref_files, est_files):
        # 读取音频文件
        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)
        
        # 如果指定了目标采样率，进行重采样
        if target_sr is not None:
            if ref_rate != target_sr:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_rate, target_sr=target_sr)
            if est_rate != target_sr:
                est_audio = librosa.resample(est_audio, orig_sr=est_rate, target_sr=target_sr)
        
        # 确保音频是单通道
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]
        if est_audio.ndim > 1:
            est_audio = est_audio[:, 0]
        
        # 使用DTW进行时间对齐
        D, wp = librosa.sequence.dtw(X=ref_audio, Y=est_audio, metric='euclidean')
        aligned_est_audio = est_audio[wp[:, 1]]
        
        # 确保对齐后的音频长度与参考音频相同
        min_len = min(len(ref_audio), len(aligned_est_audio))
        ref_audio = ref_audio[:min_len]
        aligned_est_audio = aligned_est_audio[:min_len]
        
        # 计算SDR
        sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref_audio[None, :], aligned_est_audio[None, :], compute_permutation=False)
        sdr_scores[os.path.basename(ref_path)] = sdr[0]
        mean_score.append(sdr[0])
        # print(sdr)
        # assert 1==2
    
    return sdr_scores, np.mean(mean_score)

