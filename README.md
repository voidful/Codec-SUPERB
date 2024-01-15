# Audio Codec Benchmark

## Codec Collection:

- https://github.com/ZhangXInFD/SpeechTokenizer
- https://github.com/descriptinc/descript-audio-codec
- https://github.com/facebookresearch/encodec
- https://github.com/yangdongchao/AcademiCodec
- https://github.com/facebookresearch/AudioDec
- https://github.com/alibaba-damo-academy/FunCodec

## Criteria

### Waveform (Lower is better)

L1Loss in waveform

### Mel Distance (Lower is better)

The Mel Distance is the distance between the log mel spectrograms of the reconstructed and ground truth waveforms.

### STFT Distance (Lower is better)

This metric calculates the distance between the log magnitude spectrograms of the reconstructed and ground truth
waveforms, using window lengths of [2048, 512], and is better at capturing fidelity in higher frequencies compared to
the Mel Distance.

### PESQ (Higher is better)

PESQ is an intrusive perceptual quality metric for automated assessment of the speech quality. We adopt ITU-T P.862.2 (wideband).

### STOI (Higher is better)

STOI is an intrusive perceptual quality metric that assesses audio quality based on the intelligibility of the
reconstructed speech.

