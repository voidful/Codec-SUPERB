# Audio Codec Benchmark

## Codec Collection:

- https://github.com/ZhangXInFD/SpeechTokenizer
- https://github.com/descriptinc/descript-audio-codec
- https://github.com/facebookresearch/encodec
- https://github.com/yangdongchao/AcademiCodec
- https://github.com/facebookresearch/AudioDec
- https://github.com/alibaba-damo-academy/FunCodec
- https://github.com/mct10/RepCodec

## Criteria

### Signal-to-Noise Ratio (SNR)

The Signal-to-Noise Ratio is derived from the power difference between the signal and noise, with higher values
indicating better audio quality.

### ViSQOL

ViSQOL is an intrusive perceptual quality metric that assesses audio quality based on spectral similarity to the ground
truth, providing a mean opinion score.

### Mel Distance

The Mel Distance is the distance between the log mel spectrograms of the reconstructed and ground truth waveforms.

### STFT Distance

This metric calculates the distance between the log magnitude spectrograms of the reconstructed and ground truth
waveforms, using window lengths of [2048, 512], and is better at capturing fidelity in higher frequencies compared to
the Mel Distance.

### Scale-invariant Source-to-Distortion Ratio (SI-SDR)

SI-SDR measures the distance between waveforms in a way similar to the Signal-to-Noise Ratio but is invariant to scale
differences, providing insights into the quality of the phase reconstruction of the audio.
