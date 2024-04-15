import os
import argparse
import glob
import sys
import librosa
# 
sys.path.append('visqol/visqol_lib_py')
import visqol_lib_py
import visqol_config_pb2
import similarity_result_pb2
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.transforms import Resample # Resampling
import numpy as np
import tempfile
import soundfile as sf

VISQOLMANAGER = visqol_lib_py.VisqolManager()
VISQOLMANAGER.Init(visqol_lib_py.FilePath( \
    'visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite'), \
    True, False, 60, True)

def visqol_audio(ref_folder, est_folder, sr=16000):
    ref_files = sorted(glob.glob(os.path.join(ref_folder, '*.wav')))
    est_files = sorted(glob.glob(os.path.join(est_folder, '*.wav')))
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    visqol_scores = {}
    mean_score = []
    for ref_path, est_path in zip(ref_files, est_files):
        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)
        
        # 如果指定了目标采样率，进行重采样
        if sr is not None:
            if ref_rate != sr:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_rate, target_sr=sr)
            if est_rate != sr:
                est_audio = librosa.resample(est_audio, orig_sr=est_rate, target_sr=sr)
        # 确保音频是单通道
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]
        if est_audio.ndim > 1:
            est_audio = est_audio[:, 0]
        ests = torch.from_numpy(est_audio)
        refs = torch.from_numpy(ref_audio)
        ests = ests.view(-1, ests.shape[-1])
        refs = refs.view(-1, refs.shape[-1])
        outs = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for curinx in range(ests.shape[0]):
                sf.write("{}/est_{:07d}.wav".format(tmpdirname,curinx),ests[curinx].detach().cpu().numpy(),sr)
                sf.write("{}/ref_{:07d}.wav".format(tmpdirname,curinx),refs[curinx].detach().cpu().numpy(),sr)
                out = VISQOLMANAGER.Run( \
                    visqol_lib_py.FilePath("{}/ref_{:07d}.wav".format(tmpdirname,curinx)), \
                    visqol_lib_py.FilePath("{}/est_{:07d}.wav".format(tmpdirname,curinx)))
                outs.append(out.moslqo)
        visqol_scores[os.path.basename(ref_path)] = np.mean(outs)
        mean_score.append(np.mean(outs))
    # return torch.Tensor([np.mean(outs)]).to(ests.device)

    return visqol_scores, np.mean(mean_score)

