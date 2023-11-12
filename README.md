# Audio Codec Benchmark

## Codec Collection:

- https://github.com/ZhangXInFD/SpeechTokenizer
- https://github.com/descriptinc/descript-audio-codec
- https://github.com/facebookresearch/encodec
- https://github.com/yangdongchao/AcademiCodec
- https://github.com/facebookresearch/AudioDec
- https://github.com/alibaba-damo-academy/FunCodec

## Criteria

### Waveform

L1Loss in waveform

### ViSQOL

ViSQOL is an intrusive perceptual quality metric that assesses audio quality based on spectral similarity to the ground
truth, providing a mean opinion score.

### Mel Distance

The Mel Distance is the distance between the log mel spectrograms of the reconstructed and ground truth waveforms.

### STFT Distance

This metric calculates the distance between the log magnitude spectrograms of the reconstructed and ground truth
waveforms, using window lengths of [2048, 512], and is better at capturing fidelity in higher frequencies compared to
the Mel Distance.
