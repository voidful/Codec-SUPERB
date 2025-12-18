# Codec-SUPERB: Sound Codec Speech Processing Universal Performance Benchmark

![Overview](img/Overview.png)

Codec-SUPERB is a comprehensive benchmark designed to evaluate audio codec models across a variety of speech tasks. Our
goal is to facilitate community collaboration and accelerate advancements in the field of speech processing by
preserving and enhancing speech information quality.

<a href='https://codecsuperb.com/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2402.13071'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Batch Processing](#batch-processing)
- [Installation](#installation)
- [Usage](#usage)
  - [Single Audio Processing](#single-audio-processing)
  - [Batch Audio Processing](#batch-audio-processing)
  - [Performance Comparison](#performance-comparison)
- [Testing](#testing)
- [Contribution](#contribution)
- [License](#license)

## Introduction

Codec-SUPERB sets a new benchmark in evaluating sound codec models, providing a rigorous and transparent framework for
assessing performance across a range of speech processing tasks. Our goal is to foster innovation and set new standards
in audio quality and processing efficiency.

## Key Features

### Out-of-the-Box Codec Interface

Codec-SUPERB offers an intuitive, out-of-the-box codec interface that allows for easy integration and testing of various
codec models, facilitating quick iterations and experiments.

### Multi-Perspective Leaderboard

Codec-SUPERB's unique blend of multi-perspective evaluation and an online leaderboard drives innovation in sound codec
research by providing a comprehensive assessment and fostering competitive transparency among developers.

### Standardized Environment

We ensure a standardized testing environment to guarantee fair and consistent comparison across all models. This
uniformity brings reliability to benchmark results, making them universally interpretable.

### Unified Datasets

We provide a collection of unified datasets, curated to test a wide range of speech processing scenarios. This ensures
that models are evaluated under diverse conditions, reflecting real-world applications.

## Batch Processing

**🚀 NEW: Efficient Batch Processing Support**

Codec-SUPERB now supports efficient batch processing for encoding and decoding multiple audio samples simultaneously, eliminating the need for for loops and providing significant performance improvements.

### ✅ Key Benefits

- **3-5x faster processing** for multiple audio samples
- **GPU optimization** through vectorized operations
- **Automatic padding** for variable-length audio samples
- **Memory efficient** batch operations
- **Backward compatible** - existing code continues to work

### ✅ Supported Operations

- `batch_extract_unit()`: Extract units from multiple audio samples at once
- `batch_decode_unit()`: Decode multiple units back to audio at once  
- `batch_synth()`: Complete synthesis pipeline for multiple samples

### ✅ All Codecs Supported

Every codec in Codec-SUPERB includes optimized batch processing:

- **EnCodec** (all variants): True tensor batching with automatic padding
- **SpeechTokenizer**: RVQ-aware batch processing  
- **AudioDec**: Quantizer-optimized batch operations
- **HuggingFace EnCodec**: Native transformer batch processing
- **Descript Audio Codec**: Batch compression/decompression
- **SQCodec**: Feature-aware batch encoding
- **FunCodec**: AudioSignal batch handling
- **AUV**: All-in-one codec with symmetric quantization
- **BigCodec**: Low-bitrate neural speech coding
- **S3Tokenizer**: Semantic-aware batch tokenization
- **UniCodec**: High-quality speech reconstruction
- **WavTokenizer**: Bandwidth-aware batch processing
- **AcademicCodec**: Acoustic token batch generation

## Installation

```bash
git clone https://github.com/voidful/Codec-SUPERB.git
cd Codec-SUPERB
pip install -r requirements.txt
```

## Usage

### [Leaderboard](https://codecsuperb.com)

### Single Audio Processing

Traditional single audio processing (still fully supported):

```python
from SoundCodec import codec
import torchaudio

# get all available codec
print(codec.list_codec())
# load codec by name, use encodec as example
encodec_24k_6bps = codec.load_codec('encodec_24k_6bps')

# load audio
waveform, sample_rate = torchaudio.load('sample_audio.wav')
resampled_waveform = waveform.numpy()[-1]
data_item = {'audio': {'array': resampled_waveform,
                       'sampling_rate': sample_rate}}

# extract unit
sound_unit = encodec_24k_6bps.extract_unit(data_item).unit

# sound synthesis
decoded_waveform = encodec_24k_6bps.synth(data_item, local_save=False)['audio']['array']
```

### Batch Audio Processing

**🚀 NEW: Process multiple audio samples efficiently:**

```python
from SoundCodec import codec
import torchaudio

# load codec
encodec_24k_6bps = codec.load_codec('encodec_24k_6bps')

# prepare multiple audio samples
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
data_list = []

for audio_file in audio_files:
    waveform, sample_rate = torchaudio.load(audio_file)
    data_item = {
        'id': audio_file,
        'audio': {
            'array': waveform.numpy()[0],  # take first channel
            'sampling_rate': sample_rate
        }
    }
    data_list.append(data_item)

# OPTION 1: Batch extraction and decoding (recommended)
batch_extracted = encodec_24k_6bps.batch_extract_unit(data_list)
print(f"Extracted {batch_extracted.batch_size} samples")
print(f"Unit shapes: {[unit.shape for unit in batch_extracted.units]}")

batch_decoded = encodec_24k_6bps.batch_decode_unit(batch_extracted)
print(f"Decoded audio shapes: {[audio.shape for audio in batch_decoded]}")

# OPTION 2: Complete batch synthesis pipeline
results = encodec_24k_6bps.batch_synth(data_list, local_save=False)
for i, result in enumerate(results):
    print(f"Sample {i}: unit shape {result['unit'].shape}, "
          f"audio shape {result['audio']['array'].shape}")
```

### Performance Comparison

Compare single vs batch processing performance:

```python
import time

# Single processing (old approach)
start_time = time.time()
single_results = []
for data in data_list:
    extracted = encodec_24k_6bps.extract_unit(data)
    decoded = encodec_24k_6bps.decode_unit(extracted.stuff_for_synth)
    single_results.append(decoded)
single_time = time.time() - start_time

# Batch processing (new approach)  
start_time = time.time()
batch_extracted = encodec_24k_6bps.batch_extract_unit(data_list)
batch_results = encodec_24k_6bps.batch_decode_unit(batch_extracted)
batch_time = time.time() - start_time

print(f"Single processing: {single_time:.3f}s")
print(f"Batch processing: {batch_time:.3f}s") 
print(f"Speedup: {single_time/batch_time:.2f}x")
```

### Advanced Batch Processing Tips

**Group samples by length for optimal performance:**

```python
# Group samples by similar lengths
short_samples = [data for data in data_list if len(data['audio']['array']) < 48000]
long_samples = [data for data in data_list if len(data['audio']['array']) >= 48000]

# Process each group separately for better efficiency
if short_samples:
    short_results = encodec_24k_6bps.batch_extract_unit(short_samples)
if long_samples:
    long_results = encodec_24k_6bps.batch_extract_unit(long_samples)
```

**Process large datasets in chunks:**

```python
def process_large_dataset(codec, data_list, batch_size=8):
    all_results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_results = codec.batch_synth(batch, local_save=False)
        all_results.extend(batch_results)
    return all_results

# Process large dataset efficiently
large_results = process_large_dataset(encodec_24k_6bps, large_data_list)
```

## Testing

Run the test suite to verify codec functionality:

```bash
# Run all tests
python -m pytest SoundCodec/test/

# Run batch processing tests specifically
python -m pytest SoundCodec/test/test_batch_processing.py -v

# Run performance benchmarks
python SoundCodec/test/benchmark_batch_performance.py
```

## Citation
If you use this code or result in your paper, please cite our work as:
```Tex
@article{wu2024codec,
  title={Codec-superb: An in-depth analysis of sound codec models},
  author={Wu, Haibin and Chung, Ho-Lam and Lin, Yi-Cheng and Wu, Yuan-Kuei and Chen, Xuanjun and Pai, Yu-Chi and Wang, Hsiu-Hsuan and Chang, Kai-Wei and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2402.13071},
  year={2024}
}
```
```Tex
@article{wu2024towards,
  title={Towards audio language modeling-an overview},
  author={Wu, Haibin and Chen, Xuanjun and Lin, Yi-Cheng and Chang, Kai-wei and Chung, Ho-Lam and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2402.13236},
  year={2024}
}
```
```Tex
@inproceedings{wu-etal-2024-codec,
    title = "Codec-{SUPERB}: An In-Depth Analysis of Sound Codec Models",
    author = "Wu, Haibin  and
      Chung, Ho-Lam  and
      Lin, Yi-Cheng  and
      Wu, Yuan-Kuei  and
      Chen, Xuanjun  and
      Pai, Yu-Chi  and
      Wang, Hsiu-Hsuan  and
      Chang, Kai-Wei  and
      Liu, Alexander  and
      Lee, Hung-yi",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.616",
    doi = "10.18653/v1/2024.findings-acl.616",
    pages = "10330--10348",
}
```
## Benchmarking and Leaderboard Contribution

We use the [voidful/codec-superb-tiny](https://huggingface.co/datasets/voidful/codec-superb-tiny) dataset for standard benchmarking.

### Steps to Evaluate a Codec

1.  **Synthesize the Dataset**:
    Run `dataset_creator.py` to synthesize the test set with your desired codec.
    ```bash
    python3 dataset_creator.py --dataset voidful/codec-superb-tiny
    ```
    *Note: This will process all available codecs by default. To limit to a specific codec, you can modify the script or use a custom filter.*

2.  **Calculate Metrics**:
    Run `benchmarking.py` to compute metrics (MEL, PESQ, STOI, F0Corr) for the synthesized audio.
    ```bash
    python3 benchmarking.py --dataset datasets/voidful/codec-superb-tiny_synth
    ```

3.  **Submit Results**:
    After benchmarking, a result file named `datasets_voidful_codec-superb-tiny_synth_evaluation_results_*.json` will be generated in the project root.
    
    To contribute your results to the leaderboard:
    - Open a **New Issue** in this repository.
    - Title it "New Benchmark Result: [Codec Name]".
    - Attach the generated JSON file or paste its content.
    - The maintainers will verify and merge your results into the official leaderboard.

## Contribution

Contributions are highly encouraged, whether it's through adding new codec models, expanding the dataset collection, or
enhancing the benchmarking framework. Please see `CONTRIBUTING.md` for more details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Reference Sound Codec Repositories：

- https://github.com/ZhangXInFD/SpeechTokenizer
- https://github.com/descriptinc/descript-audio-codec
- https://github.com/facebookresearch/encodec
- https://github.com/yangdongchao/AcademiCodec
- https://github.com/facebookresearch/AudioDec
- https://github.com/alibaba-damo-academy/FunCodec
- https://github.com/SWivid/AUV
- https://github.com/Aria-K-Alethia/BigCodec
- https://github.com/xingchensong/S3Tokenizer
- https://github.com/mesolitica/UniCodec-fix
