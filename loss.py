import typing
from typing import List
import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
import torchaudio
from pesq import pesq
import numpy


class PESQ(nn.Module):
    def __init__(self, band_type: str = "wb"):
        self.band_type = band_type
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal, sample_rate: int = 16000):

        references = x.resample(sample_rate).audio_data.cpu().detach().numpy()
        estimates = y.resample(sample_rate).audio_data.cpu().detach().numpy()

        if len(references.shape) == 1 and len(estimates.shape) == 1:
            references = numpy.expand_dims(references, axis=0)
            estimates = numpy.expand_dims(references, axis=0)

        pesq_scores = []
        for ref, est in zip(references, estimates):
            pesq_mos = pesq(sample_rate, ref, est, self.band_type)
            pesq_scores.append(pesq_mos)

        return sum(pesq_scores) / len(pesq_scores)


class STOI(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal, sample_rate: int):

        references = x.audio_data.cpu().detach().numpy()
        estimates = y.audio_data.cpu().detach().numpy()

        if len(references.shape) == 1 and len(estimates.shape) == 1:
            references = numpy.expand_dims(references, axis=0)
            estimates = numpy.expand_dims(references, axis=0)

        stoi_scores = []
        for ref, est in zip(references, estimates):
            stoi_score = pesq(ref, est, sample_rate)
            stoi_scores.append(stoi_score)

        return sum(pesq_scores) / len(pesq_scores)


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
        return sdr


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
        return self.weight * snr


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
        x_hat[:, 0 : (int(K / 2) + 1)]
    )  # take clean single-sided spectrum
    y_hat = np.transpose(
        y_hat[:, 0 : (int(K / 2) + 1)]
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
            :, (m - N + 1) : (m + 1)
        ]  # region with length N of clean TF-units for all j
        Y_seg = Y[
            :, (m - N + 1) : (m + 1)
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
