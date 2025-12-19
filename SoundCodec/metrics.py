import typing
from typing import List

import librosa
import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
from scipy.linalg import norm
from scipy import signal
from pesq import pesq
import numpy as np
import parselmouth
from torchmetrics import PearsonCorrCoef


class PESQ(nn.Module):
    def __init__(self, band_type: str = "wb"):
        self.band_type = band_type
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal, sample_rate: int = 16000):
        try:
            references = x.resample(sample_rate).audio_data.squeeze(0).cpu().numpy()
            estimates = y.resample(sample_rate).audio_data.squeeze(0).cpu().numpy()

            if len(references.shape) == 1:
                references = np.expand_dims(references, axis=0)
                estimates = np.expand_dims(estimates, axis=0)

            pesq_scores = [pesq(sample_rate, ref, est, self.band_type) for ref, est in zip(references, estimates)]
            return float(np.mean(pesq_scores))
        except Exception:
            return 1.0


class STOI(nn.Module):
    def forward(self, x: AudioSignal, y: AudioSignal, sample_rate: int):
        try:
            references = x.audio_data.squeeze(0).cpu().numpy()
            estimates = y.audio_data.squeeze(0).cpu().numpy()

            if len(references.shape) == 1:
                references = np.expand_dims(references, axis=0)
                estimates = np.expand_dims(estimates, axis=0)

            stoi_scores = [stoi(ref, est, sample_rate) for ref, est in zip(references, estimates)]
            stoi_scores_clipped = np.clip(stoi_scores, 0.0, 1.0)
            if len(stoi_scores_clipped) == 0 or np.isnan(np.mean(stoi_scores_clipped)):
                return 0.0
            return float(np.mean(stoi_scores_clipped))
        except Exception:
            return 0.0


class L1Loss(nn.L1Loss):
    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: AudioSignal, y: AudioSignal):
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    def __init__(
            self,
            scaling: int = True,
            reduction: str = "mean",
            zero_mean: int = True,
            clip_min: int = None,
            weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references ** 2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr.item()


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
            self,
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(
            self,
            n_mels: List[int] = [150, 80],
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            mel_fmin: List[float] = [0.0, 0.0],
            mel_fmax: List[float] = [None, None],
            window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
                self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class SignalToNoiseRatioLoss(nn.Module):
    def __init__(self, attribute="audio_data", weight=1.0):
        super(SignalToNoiseRatioLoss, self).__init__()
        self.attribute = attribute
        self.weight = weight

    def forward(self, x: AudioSignal, y: AudioSignal):
        x_audio = getattr(x, self.attribute)
        y_audio = getattr(y, self.attribute)

        noise = x_audio - y_audio
        snr = 10 * torch.log10(torch.sum(x_audio ** 2) / torch.sum(noise ** 2))
        return (self.weight * snr).item()


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
                np.sum(np.square(Y_seg), axis=1, keepdims=True) + 1e-10,
            )
        )
        # obtain \alpha*Y_j(n) from Eq.(2) [1]
        aY_seg = np.multiply(Y_seg, alpha)
        # apply clipping from Eq.(3)
        Y_prime = np.minimum(aY_seg, X_seg + X_seg * c)
        # obtain correlation coeffecient from Eq.(4) [1]
        d_interm[m - N + 1] = taa_corr(X_seg, Y_prime) / J

    if len(d_interm) == 0:
        return 0.0
    d = (
        d_interm.mean()
    )  # combine all intermediate intelligibility measures as in Eq.(4) [1]
    return float(d)


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

    msk = 20 * np.log10(np.divide(norm(np.multiply(x[jj_list], w), axis=1), np.sqrt(N)) + 1e-10)

    msk = (msk - np.max(msk) + dyrange) > 0
    count = 0

    x_sil = np.zeros(np.size(x))
    y_sil = np.zeros(np.size(y))

    jj_o = [0]
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


def taa_corr(x, y):
    """
    RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
    vectors x and y. Gives same results as 'corr' from statistics toolbox.
    """
    xn = np.subtract(x, np.mean(x, axis=1, keepdims=True))
    xn = np.divide(xn, norm(xn, axis=1, keepdims=True) + 1e-10)
    yn = np.subtract(y, np.mean(y, axis=1, keepdims=True))
    yn = np.divide(yn, norm(yn, axis=1, keepdims=True) + 1e-10)
    rho = np.trace(np.matmul(xn, np.transpose(yn)))

    return rho


def get_metrics(signal, recons):
    with torch.no_grad():
        if isinstance(signal, str):
            signal = AudioSignal(signal)
        if isinstance(recons, str):
            recons = AudioSignal(recons)

        x = signal
        y = recons.clone().resample(x.sample_rate)

        metrics = {}
        metric_functions = {
            "mel": lambda: mel_loss(x, y).cpu().item(),
            "stft": lambda: stft_loss(x, y).cpu().item(),
            "waveform": lambda: waveform_loss(x, y).cpu().item(),
            "pesq": lambda: pesqfn(x, y),
            "pesq-wb": lambda: pesqfn(x, y),
            "stoi": lambda: stoifn(x, y, x.sample_rate),
            "f0corr": lambda: f0corr(x, y),
        }

        for metric_name, metric_func in metric_functions.items():
            try:
                metrics[metric_name] = metric_func()
            except Exception:
                metrics[metric_name] = np.nan

        return metrics


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class F0CorrLoss(torch.nn.Module):
    def __init__(self, sample_rate=22050, hop_length=256, f0_min=50, f0_max=1100, pitch_bin=256, pitch_min=50,
                 pitch_max=1100, need_mean=True, method="dtw"):
        super(F0CorrLoss, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.pitch_bin = pitch_bin
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.need_mean = need_mean
        self.method = method
        self.pearson = PearsonCorrCoef()

    def f0_to_coarse(self, f0, pitch_bin, pitch_min, pitch_max):
        f0_mel_min = 1127 * np.log(1 + pitch_min / 700)
        f0_mel_max = 1127 * np.log(1 + pitch_max / 700)
        is_torch = isinstance(f0, torch.Tensor)
        f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (pitch_bin - 2) / (
                f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > pitch_bin - 1] = pitch_bin - 1
        f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int32)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def get_f0_features_using_parselmouth(self, audio, cfg, speed=1):
        hop_size = int(np.round(cfg.hop_size * speed))

        # Calculate the time step for pitch extraction
        time_step = hop_size / cfg.sample_rate * 1000

        f0 = (
            parselmouth.Sound(audio, cfg.sample_rate)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=cfg.f0_min,
                pitch_ceiling=cfg.f0_max,
            )
            .selected_array["frequency"]
        )

        # Pad the pitch to the mel_len
        # pad_size = (int(len(audio) // hop_size) - len(f0) + 1) // 2
        # f0 = np.pad(f0, [[pad_size, mel_len - len(f0) - pad_size]], mode="constant")

        # Get the coarse part
        pitch_coarse = self.f0_to_coarse(f0, cfg.pitch_bin, cfg.f0_min, cfg.f0_max)
        return f0, pitch_coarse

    def get_cents(self, f0_hz):
        """
        F_{cent} = 1200 * log2 (F/440)

        Reference:
            APSIPA'17, Perceptual Evaluation of Singing Quality
        """
        voiced_f0 = f0_hz[f0_hz != 0]
        return 1200 * np.log2(voiced_f0 / 440)

    def get_pitch_sub_median(self, f0_hz):
        """
        f0_hz: (,T)
        """
        f0_cent = self.get_cents(f0_hz)
        return f0_cent - np.median(f0_cent)

    def process_audio(self, audio):
        # Initialize config
        cfg = JsonHParams()
        cfg.sample_rate = self.sample_rate
        cfg.hop_size = self.hop_length
        cfg.f0_min = self.f0_min
        cfg.f0_max = self.f0_max
        cfg.pitch_bin = self.pitch_bin
        cfg.pitch_max = self.pitch_max
        cfg.pitch_min = self.pitch_min

        # Extract F0
        try:
            f0 = self.get_f0_features_using_parselmouth(audio.audio_data[0].cpu().detach().numpy(), cfg)[0]
        except Exception:
            return None

        # Subtract mean if needed
        if self.need_mean:
            if f0 is None or len(f0[f0 != 0]) == 0:
                return None
            f0 = torch.from_numpy(f0)
            f0 = self.get_pitch_sub_median(f0).numpy()

        return f0

    def forward(self, audio_ref, audio_deg):
        f0_ref = self.process_audio(audio_ref)
        f0_deg = self.process_audio(audio_deg)

        if f0_ref is None or f0_deg is None:
            return 0.0

        # Avoid silence
        min_length = min(len(f0_ref), len(f0_deg))
        if min_length <= 1:
            return 0.0

        # F0 length alignment
        if self.method == "cut":
            length = min(len(f0_ref), len(f0_deg))
            f0_ref = f0_ref[:length]
            f0_deg = f0_deg[:length]
        elif self.method == "dtw":
            try:
                _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
                f0_ref = np.array([f0_ref[gt_index] for gt_index, _ in wp])
                f0_deg = np.array([f0_deg[pred_index] for _, pred_index in wp])
            except Exception:
                return torch.tensor(0.0)

        # Convert to tensor and calculate Pearson correlation coefficient
        f0_ref = torch.from_numpy(f0_ref).float()
        f0_deg = torch.from_numpy(f0_deg).float()
        try:
            res = self.pearson(f0_ref, f0_deg)
            if torch.isnan(res):
                return 0.0
            return res.item()
        except Exception:
            return 0.0


waveform_loss = L1Loss()
stft_loss = MultiScaleSTFTLoss()
mel_loss = MelSpectrogramLoss()
sisdr_loss = SISDRLoss()
snr_loss = SignalToNoiseRatioLoss()
pesqfn = PESQ()
stoifn = STOI()
f0corr = F0CorrLoss()
