# Codec-SUPERB: Sound Codec Speech Processing Universal Performance Benchmark

<div align="center">

![Overview](img/Overview.png)

[![Project Page](https://img.shields.io/badge/Project-Page-Green?style=for-the-badge)](https://codecsuperb.com/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red?style=for-the-badge)](https://arxiv.org/abs/2402.13071)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**A comprehensive benchmark evaluating audio codec models across diverse speech processing tasks.**

</div>

---

## 📖 Introduction

**Codec-SUPERB** sets a new standard for evaluating sound codec models. We provide a rigorous and transparent framework for assessing speech quality and information preservation across various downstream tasks. Our goal is to foster innovation and facilitate community collaboration in the field of neural audio coding.

---

## ✨ Key Features

* 🚀 **Out-of-the-Box Interface**: Intuitive API for easy integration and rapid experimentation with diverse codec models.
* 📊 **Multi-Perspective Leaderboard**: Comprehensive assessment across various speech processing dimensions with rankings for competitive transparency.
* 🏗️ **Standardized Environment**: Ensures fair and consistent comparisons by using uniform testing conditions for all models.
* 📚 **Unified Datasets**: Curated collection of datasets testing a wide range of real-world speech processing scenarios.
* ⚡ **Batch Processing Support**: Highly optimized batch encoding/decoding for significant performance speedups.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/voidful/Codec-SUPERB.git
cd Codec-SUPERB

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### List and Load Codecs

```python
from SoundCodec import codec

# List all available codecs
print(codec.list_codec())

# Load a specific codec
model = codec.load_codec('encodec_24k_6bps')
```

### Single Audio Processing

```python
import torchaudio

# Load audio
waveform, sample_rate = torchaudio.load('sample_audio.wav')
data_item = {'audio': {'array': waveform.numpy()[-1], 'sampling_rate': sample_rate}}

# Extract discrete units
sound_unit = model.extract_unit(data_item).unit

# Reconstruct audio
reconstructed = model.synth(data_item, local_save=False)['audio']['array']
```

---

## ⚡ Advanced Usage: Batch Processing

Codec-SUPERB supports efficient batch operations, typically providing **3-5x performance improvement** on GPU.

```python
# Prepare multiple samples
data_list = [
    {'audio': {'array': wave1, 'sampling_rate': 16000}},
    {'audio': {'array': wave2, 'sampling_rate': 16000}}
]

# Option 1: Batch extraction and decoding (Recommended)
batch_extracted = model.batch_extract_unit(data_list)
batch_decoded = model.batch_decode_unit(batch_extracted)

# Option 2: Complete batch synthesis pipeline
results = model.batch_synth(data_list, local_save=False)
```

> [!TIP]
> Grouping samples by similar lengths can further optimize batch processing efficiency.

---

## 🎯 Benchmarking & Leaderboard

Follow these steps to evaluate your codec and contribute to the [Official Leaderboard](https://codecsuperb.com).

### 1. Synthesize the Test Set

Use the [voidful/codec-superb-tiny](https://huggingface.co/datasets/voidful/codec-superb-tiny) dataset:

```bash
PYTHONPATH=. python3 scripts/dataset_creator.py --dataset voidful/codec-superb-tiny
```

### 2. Calculate Metrics

Compute standard metrics (MEL, PESQ, STOI, F0Corr):

```bash
# Benchmark all codecs
PYTHONPATH=. python3 scripts/benchmarking.py --dataset datasets/voidful/codec-superb-tiny_synth

# Benchmark only specific codec(s)
PYTHONPATH=. python3 scripts/benchmarking.py \
    --dataset datasets/voidful/codec-superb-tiny_synth \
    --models llmcodec
```

### 3. Submit Results

1. Locate the generated JSON file: `datasets_voidful_codec-superb-tiny_synth_evaluation_results_*.json`.
2. Open a **New Issue** in this repository titled `New Benchmark Result: [Codec Name]`.
3. Attach the JSON file or paste its content.

---

## 🛡️ Encode-Only Codec Support

Certain codecs (e.g., `s3tokenizer`) focus on tokenization and do not support reconstruction. Codec-SUPERB handles these automatically:

* **Benchmarking**: Automatically skipped during reconstruction evaluation.
* **API**: Raises `NotImplementedError` if `decode_unit` is called, with clear messaging.

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest SoundCodec/test/

# Verify all codecs (Initialization & Synthesis)
PYTHONPATH=. python3 scripts/check_all_codecs.py
```

---

## 📝 Citation

If you use Codec-SUPERB in your research, please cite:

```bibtex
@inproceedings{wu-etal-2024-codec,
    title = "Codec-{SUPERB}: An In-Depth Analysis of Sound Codec Models",
    author = "Wu, Haibin and Chung, Ho-Lam and Lin, Yi-Cheng and Wu, Yuan-Kuei and Chen, Xuanjun and Pai, Yu-Chi and Wang, Hsiu-Hsuan and Chang, Kai-Wei and Liu, Alexander and Lee, Hung-yi",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-acl.616",
    doi = "10.18653/v1/2024.findings-acl.616",
    pages = "10330--10348",
}
```

---

## 🤝 Contribution & License

Contributions are highly encouraged! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
This project is licensed under the **MIT License**.

---

<div align="center">
Developed with ❤️ by the Codec-SUPERB Team
</div>
