# Codec-SUPERB: Sound Codec Speech Processing Universal Performance Benchmark

![Overview](img/Overview.png)

Codec-SUPERB is a comprehensive benchmark designed to evaluate audio codec models across a variety of speech tasks. Our
goal is to facilitate community collaboration and accelerate advancements in the field of speech processing by
preserving and enhancing speech information quality.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
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

## Installation

```bash
git clone https://github.com/voidful/Codec-SUPERB.git
cd Codec-SUPERB
pip install -r requirements.txt
```

## Usage

### [Leaderboard](https://codecsuperb.com)

### Out of the Box Codec Interface

```python
from SoundCodec import codec
import torchaudio

# get all available codec
print(codec.list_codec())
# load codec by name, use encodec as example
encodec_24k_6bps = codec.load_codec('encodec_24k_6bps')

# load audio
waveform, sample_rate = torchaudio.load('sample audio')
resampled_waveform = waveform.numpy()[-1]
data_item = {'audio': {'array': resampled_waveform,
                       'sampling_rate': sample_rate}}

# extract unit
sound_unit = encodec_24k_6bps.extract_unit(data_item).unit

# sound synthesis
decoded_waveform = encodec_24k_6bps.synth(sound_unit, local_save=False)['audio']['array']
```

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
