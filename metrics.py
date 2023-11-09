from audiotools import AudioSignal
from audiotools import metrics

from loss import L1Loss, MultiScaleSTFTLoss, MelSpectrogramLoss, SISDRLoss, SpectrogramErrorLoss, SignalToNoiseRatioLoss

waveform_loss = L1Loss()
stft_loss = MultiScaleSTFTLoss()
mel_loss = MelSpectrogramLoss()
sisdr_loss = SISDRLoss()
snr_loss = SignalToNoiseRatioLoss()

def get_metrics(signal, recons):
    if isinstance(signal, str):
        signal = AudioSignal(signal)
    if isinstance(recons, str):
        recons = AudioSignal(recons)

    output = {}
    x = signal
    y = recons.clone().resample(x.sample_rate)
    output.update(
        {
            f"mel": mel_loss(x, y).cpu().detach().item(),
            f"stft": stft_loss(x, y).cpu().detach().item(),
            f"waveform": waveform_loss(x, y).cpu().detach().item(),
            f"visqol-audio": metrics.quality.visqol(x, y).cpu().detach().item(),
            f"visqol-speech": metrics.quality.visqol(x, y, "speech").cpu().detach().item()
        }
    )
    return output
