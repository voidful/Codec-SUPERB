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

### Compression Ratio (CR)
The Compression Ratio is calculated by dividing the compressed file size by the original uncompressed file size, with lower values being preferable.

### Average Bitrate
The Average Bitrate is computed as the product of the compression ratio and the original bitrate.

### Spectrogram Error
This metric represents the direct difference between the power spectrograms of the original and compressed files, with lower values indicating better performance.

### Weighted Spectrogram Error
Similar to the Spectrogram Error, but with the difference calculated between A-weighted power spectrograms. Lower values are better.

### Signal-to-Noise Ratio (SNR)
The Signal-to-Noise Ratio is derived from the power difference between the signal and noise, with higher values indicating better audio quality.

### Weighted SNR
This is a variant of the Signal-to-Noise Ratio, where the noise is calculated from the A-weighted signal power difference. Higher values indicate better audio quality.

### ViSQOL
ViSQOL is an intrusive perceptual quality metric that assesses audio quality based on spectral similarity to the ground truth, providing a mean opinion score.

### Mel Distance
The Mel Distance is the distance between the log mel spectrograms of the reconstructed and ground truth waveforms, configured as described in section 3.5 of the referenced document.

### STFT Distance
This metric calculates the distance between the log magnitude spectrograms of the reconstructed and ground truth waveforms, using window lengths of [2048, 512], and is better at capturing fidelity in higher frequencies compared to the Mel Distance.

### Scale-invariant Source-to-Distortion Ratio (SI-SDR)
SI-SDR measures the distance between waveforms in a way similar to the Signal-to-Noise Ratio but is invariant to scale differences, providing insights into the quality of the phase reconstruction of the audio.

### Bitrate Efficiency
Bitrate Efficiency is calculated as the sum of the entropy (in bits) of each codebook applied on a large test set, divided by the total number of bits across all codebooks, aiming for 100% for efficient bitrate utilization.
