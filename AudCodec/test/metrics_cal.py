import unittest
import numpy as np
import torch
from metrics import PESQ, STOI, L1Loss, SISDRLoss, MultiScaleSTFTLoss, MelSpectrogramLoss, SignalToNoiseRatioLoss, \
    F0CorrLoss
from audiotools import AudioSignal


def generate_sine_wave(frequency, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = np.sin(frequency * t * 2 * np.pi)
    return sine_wave


def generate_noise(sample_rate, duration):
    return np.random.randn(int(sample_rate * duration))


class TestAudioMetrics(unittest.TestCase):

    def setUp(self):
        # Basic setup for each test
        self.sample_rate = 16000
        self.duration = 1  # 1 second
        self.clean_signal = generate_sine_wave(440, self.sample_rate, self.duration)  # A4 note
        self.noisy_signal = self.clean_signal + generate_noise(self.sample_rate, self.duration) * 0.5

        # Convert numpy arrays to AudioSignal
        self.clean_audio = AudioSignal(self.clean_signal, self.sample_rate)
        self.noisy_audio = AudioSignal(self.noisy_signal, self.sample_rate)

    def test_pesq(self):
        pesq_metric = PESQ()
        score = pesq_metric(self.clean_audio, self.noisy_audio, self.sample_rate)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 4.5)

    def test_stoi(self):
        stoi_metric = STOI()
        score = stoi_metric(self.clean_audio, self.noisy_audio, self.sample_rate)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_l1_loss(self):
        l1_loss_metric = L1Loss()
        loss = l1_loss_metric(self.clean_audio, self.noisy_audio)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss, 0.0)

    def test_sisdr_loss(self):
        sisdr_metric = SISDRLoss()
        loss = sisdr_metric(self.clean_audio, self.noisy_audio)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertLess(loss, 0.0)

    def test_multiscale_stft_loss(self):
        mstft_metric = MultiScaleSTFTLoss()
        loss = mstft_metric(self.clean_audio, self.noisy_audio)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss, 0.0)

    def test_mel_spectrogram_loss(self):
        mel_loss_metric = MelSpectrogramLoss()
        loss = mel_loss_metric(self.clean_audio, self.noisy_audio)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss, 0.0)

    def test_snr_loss(self):
        snr_metric = SignalToNoiseRatioLoss()
        loss = snr_metric(self.clean_audio, self.noisy_audio)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss, 0.0)

    def test_f0_corr_loss(self):
        f0_corr_metric = F0CorrLoss()
        score = f0_corr_metric(self.clean_audio, self.noisy_audio)
        self.assertIsInstance(score, torch.Tensor)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
