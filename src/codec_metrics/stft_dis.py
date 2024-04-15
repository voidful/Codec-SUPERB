import numpy as np
import soundfile as sf
import librosa
import glob
import os
import numpy as np

def STFT_distance(ref_folder, est_folder, target_sr=16000, n_fft=2048, hop_length=512):
    """
    Calculate the Short-Time Fourier Transform (STFT) distance between pairs of reference and estimated audio files
    located in the given directories, optionally resampling all files to a specified sample rate.
    
    Parameters:
        ref_folder (str): The folder path containing the reference audio files (.wav).
        est_folder (str): The folder path containing the estimated/generated audio files (.wav).
        target_sr (int, optional): The target sample rate to which all audio files will be resampled. If None, no resampling is performed.
        n_fft (int): The number of data points used in each block for the FFT. Default is 2048.
        hop_length (int): The number of audio samples between adjacent STFT columns. Default is 512.
    
    Returns:
        dict: A dictionary containing the STFT distances for each pair of audio files, with file names as keys.
    """
    # 获取所有参考音频和生成音频的路径
    ref_files = sorted(glob.glob(os.path.join(ref_folder, '*.wav')))
    est_files = sorted(glob.glob(os.path.join(est_folder, '*.wav')))
    
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    
    stft_distances = {}
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
        
        # 计算STFT
        ref_stft = librosa.stft(ref_audio, n_fft=n_fft, hop_length=hop_length)
        est_stft = librosa.stft(est_audio, n_fft=n_fft, hop_length=hop_length)
        min_len = min(ref_stft.shape[1], est_stft.shape[1])
        ref_stft = ref_stft[:,:min_len]
        est_stft = est_stft[:,:min_len]
        # 计算STFT的欧氏距离
        distance = np.linalg.norm(np.abs(ref_stft) - np.abs(est_stft))
        mean_score.append(distance)
        # 存储距离
        stft_distances[os.path.basename(ref_path)] = distance
    
    return stft_distances, np.mean(mean_score)

