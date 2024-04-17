import numpy as np
import soundfile as sf
import librosa
import glob
import os
from torch import nn
import numpy as np
from audiotools import AudioSignal

def Mel_loss(ref_folder, 
             est_folder, 
             target_sr = None,            
             n_mels = [150, 80],
             window_lengths = [2048, 512],
             loss_fn = nn.L1Loss(),
             clamp_eps = 1e-5,
             mag_weight = 1.0,
             log_weight = 1.0,
             pow = 2.0,
             match_stride = False,
             mel_fmin = [0.0, 0.0],
             mel_fmax = [None, None],
             window_type = None
            ):
    """
    Calculate the Mel Spectrogram distance between pairs of reference and estimated audio files
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
    
    mel_losses = {}
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
        
        ref_signal = AudioSignal(ref_audio, target_sr if target_sr is not None else ref_rate)
        est_signal = AudioSignal(est_audio, target_sr if target_sr is not None else est_rate)
        
        # 计算 Mel Loss
        loss = 0.0
        for n_mel, fmin, fmax, wlen in zip(
                n_mels, mel_fmin, mel_fmax, window_lengths
        ):
            kwargs = {
                "window_length": wlen,
                "hop_length": wlen // 4,
                "window_type": window_type,
            }
            x_mels = ref_signal.mel_spectrogram(n_mel, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = est_signal.mel_spectrogram(n_mel, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += log_weight * loss_fn(
                x_mels.clamp(clamp_eps).pow(pow).log10(),
                y_mels.clamp(clamp_eps).pow(pow).log10(),
            )
            loss += mag_weight * loss_fn(x_mels, y_mels)
        
        mean_score.append(loss)
        # 存储距离
        mel_losses[os.path.basename(ref_path)] = loss
    
    return mel_losses, np.mean(mean_score)

