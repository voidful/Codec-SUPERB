import numpy as np
import soundfile as sf
import glob
import os
import numpy as np
from scipy import signal
from scipy.linalg import norm

def stoi_folder(ref_folder, est_folder):
    """
    Calculate STOI (Short-Time Objective Intelligibility) metric between pairs of reference and estimated audio files
    located in the given directories, optionally resampling all files to a specified sample rate.
    
    Parameters:
        ref_folder (str): The folder path containing the reference audio files (.wav).
        est_folder (str): The folder path containing the estimated/generated audio files (.wav).
    
    Returns:
        dict: A dictionary containing the STOI for each pair of audio files, with file names as keys.
    """
    # 获取所有参考音频和生成音频的路径
    ref_files = sorted(glob.glob(os.path.join(ref_folder, '*.wav')))
    est_files = sorted(glob.glob(os.path.join(est_folder, '*.wav')))
    
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    
    stoi_score = {}
    mean_score = []
    for ref_path, est_path in zip(ref_files, est_files):
        # 读取音频文件
        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)
        
        # 确保音频是单通道
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]
        if est_audio.ndim > 1:
            est_audio = est_audio[:, 0]
        score = stoi(ref_audio, est_audio, 10000)
        mean_score.append(score)
        stoi_score[os.path.basename(ref_path)] = score
    
    return stoi_score, np.mean(mean_score)
        



def stoi(x, y, fs_signal):
    if np.size(x) != np.size(y):
        raise ValueError("x and y should have the same length")

    # initialization, pay attention to the range of x and y(divide by 32768?)
    fs = 10000  # sample rate of proposed intelligibility measure
    N_frame = 256  # window support
    K = 512  # FFT size
    J = 15  # Number of 1/3 octave bands
    mn = 150  # Center frequency of first 1/3 octave band in Hz
    H, _ = thirdoct(fs, K, J, mn)  # Get 1/3 octave band matrix
    N = 30  # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15  # lower SDR-bound
    dyn_range = 40  # speech dynamic range

    # resample signals if other sample rate is used than fs
    if fs_signal != fs:
        x = signal.resample_poly(x, fs, fs_signal)
        y = signal.resample_poly(y, fs, fs_signal)

    # remove silent frames
    x, y = removeSilentFrames(x, y, dyn_range, N_frame, int(N_frame / 2))

    # apply 1/3 octave band TF-decomposition
    x_hat = stdft(x, N_frame, N_frame / 2, K)  # apply short-time DFT to clean speech
    y_hat = stdft(
        y, N_frame, N_frame / 2, K
    )  # apply short-time DFT to processed speech

    x_hat = np.transpose(
        x_hat[:, 0: (int(K / 2) + 1)]
    )  # take clean single-sided spectrum
    y_hat = np.transpose(
        y_hat[:, 0: (int(K / 2) + 1)]
    )  # take processed single-sided spectrum

    X = np.sqrt(
        np.matmul(H, np.square(np.abs(x_hat)))
    )  # apply 1/3 octave bands as described in Eq.(1) [1]
    Y = np.sqrt(np.matmul(H, np.square(np.abs(y_hat))))

    # loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
    d_interm = np.zeros(np.size(np.arange(N - 1, x_hat.shape[1])))
    # init memory for intermediate intelligibility measure
    c = 10 ** (-Beta / 20)
    # constant for clipping procedure

    for m in range(N - 1, x_hat.shape[1]):
        X_seg = X[
                :, (m - N + 1): (m + 1)
                ]  # region with length N of clean TF-units for all j
        Y_seg = Y[
                :, (m - N + 1): (m + 1)
                ]  # region with length N of processed TF-units for all j
        # obtain scale factor for normalizing processed TF-region for all j
        alpha = np.sqrt(
            np.divide(
                np.sum(np.square(X_seg), axis=1, keepdims=True),
                np.sum(np.square(Y_seg), axis=1, keepdims=True),
            )
        )
        # obtain \alpha*Y_j(n) from Eq.(2) [1]
        aY_seg = np.multiply(Y_seg, alpha)
        # apply clipping from Eq.(3)
        Y_prime = np.minimum(aY_seg, X_seg + X_seg * c)
        # obtain correlation coeffecient from Eq.(4) [1]
        d_interm[m - N + 1] = taa_corr(X_seg, Y_prime) / J

    d = (
        d_interm.mean()
    )  # combine all intermediate intelligibility measures as in Eq.(4) [1]
    return d


def removeSilentFrames(x, y, dyrange, N, K):
    """
    [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y are segmented with frame-length N
    and overlap K, where the maximum energy of all frames of X is determined, say X_MAX.
    X_SIL and Y_SIL are the reconstructed signals, excluding the frames, where the energy of a frame
    of X is smaller than X_MAX-RANGE
    """

    frames = np.arange(0, (np.size(x) - N), K)
    w = signal.windows.hann(N + 2)
    w = w[1: N + 1]

    jj_list = np.empty((np.size(frames), N), dtype=int)
    for j in range(np.size(frames)):
        jj_list[j, :] = np.arange(frames[j] - 1, frames[j] + N - 1)

    msk = 20 * np.log10(np.divide(norm(np.multiply(x[jj_list], w), axis=1), np.sqrt(N)))

    msk = (msk - np.max(msk) + dyrange) > 0
    count = 0

    x_sil = np.zeros(np.size(x))
    y_sil = np.zeros(np.size(y))

    for j in range(np.size(frames)):
        if msk[j]:
            jj_i = np.arange(frames[j], frames[j] + N)
            jj_o = np.arange(frames[count], frames[count] + N)
            x_sil[jj_o] = x_sil[jj_o] + np.multiply(x[jj_i], w)
            y_sil[jj_o] = y_sil[jj_o] + np.multiply(y[jj_i], w)
            count = count + 1

    x_sil = x_sil[0: jj_o[-1] + 1]
    y_sil = y_sil[0: jj_o[-1] + 1]
    return x_sil, y_sil


def thirdoct(fs, N_fft, numBands, mn):
    """
    [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
    inputs:
        FS:         samplerate
        N_FFT:      FFT size
        NUMBANDS:   number of bands
        MN:         center frequency of first 1/3 octave band
    outputs:
        A:          octave band matrix
        CF:         center frequencies
    """
    f = np.linspace(0, fs, N_fft + 1)
    f = f[0: int(N_fft / 2 + 1)]
    k = np.arange(numBands)
    cf = np.multiply(np.power(2, k / 3), mn)
    fl = np.sqrt(
        np.multiply(
            np.multiply(np.power(2, k / 3), mn),
            np.multiply(np.power(2, (k - 1) / 3), mn),
        )
    )
    fr = np.sqrt(
        np.multiply(
            np.multiply(np.power(2, k / 3), mn),
            np.multiply(np.power(2, (k + 1) / 3), mn),
        )
    )
    A = np.zeros((numBands, len(f)))

    for i in range(np.size(cf)):
        b = np.argmin((f - fl[i]) ** 2)
        fl[i] = f[b]
        fl_ii = b

        b = np.argmin((f - fr[i]) ** 2)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii:fr_ii] = 1

    rnk = np.sum(A, axis=1)
    end = np.size(rnk)
    rnk_back = rnk[1:end]
    rnk_before = rnk[0: (end - 1)]
    for i in range(np.size(rnk_back)):
        if (rnk_back[i] >= rnk_before[i]) and (rnk_back[i] != 0):
            result = i
    numBands = result + 2
    A = A[0:numBands, :]
    cf = cf[0:numBands]
    return A, cf

def stdft(x, N, K, N_fft):
    """
    X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time hanning-windowed dft of X with frame-size N,
    overlap K and DFT size N_FFT. The columns and rows of X_STDFT denote the frame-index and dft-bin index,
    respectively.
    """
    frames_size = int((np.size(x) - N) / K)
    w = signal.windows.hann(N + 2)
    w = w[1: N + 1]

    x_stdft = signal.stft(
        x,
        window=w,
        nperseg=N,
        noverlap=K,
        nfft=N_fft,
        return_onesided=False,
        boundary=None,
    )[2]
    x_stdft = np.transpose(x_stdft)[0:frames_size, :]

    return x_stdft

def taa_corr(x, y):
    """
    RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
    vectors x and y. Gives same results as 'corr' from statistics toolbox.
    """
    xn = np.subtract(x, np.mean(x, axis=1, keepdims=True))
    xn = np.divide(xn, norm(xn, axis=1, keepdims=True))
    yn = np.subtract(y, np.mean(y, axis=1, keepdims=True))
    yn = np.divide(yn, norm(yn, axis=1, keepdims=True))
    rho = np.trace(np.matmul(xn, np.transpose(yn)))

    return rho