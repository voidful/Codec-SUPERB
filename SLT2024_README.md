# Codec-SUPERB (SLT 2024 challenge)

## Environment installation
### 1.1 Application - Emotion
```shell
bash envs/emo2vec.sh
```

### 1.2 Application - Speaker verification
```shell
bash envs/speaker.sh
```

### 1.3 Application - Automatic speech recognition
```shell
bash envs/asr.sh
```

### 1.4 Application - Audio Event Classification
```shell
bash envs/aec.sh
```

### 2. Objective metrics
```shell
bash envs/metrics.sh
```

## Data download
```shell
pip install gdown
gdown 1V_uHK7JO2_o7S41KS69fI-pTCKndP3UJ
```
After `unzip` the `codec_superb_data.zip`, you can obtain following files:
```
.
├── ESC-50-master
│   ├── audio
│   └── meta
├── LibriSpeech
│   ├── BOOKS.TXT
│   ├── CHAPTERS.TXT
│   ├── LICENSE.TXT
│   ├── README.TXT
│   ├── SPEAKERS.TXT
│   ├── test-clean
│   └── test-other
├── RAVDESS
│   ├── ravdess
│   └── ravdess.txt
├── samples
│   ├── fsd50k
│   ├── esc50
│   ├── gunshot_triangulation
│   ├── crema_d
│   ├── fluent_speech_commands
│   ├── libri2Mix_test
│   ├── librispeech
│   ├── quesst
│   ├── snips_test_valid_subset
│   ├── voxceleb1
│   └── vox_lingua_top10
└── vox1_test_wav
    └── wav
```
The resynthesised audio files, should follow the same structure for evaluation (`run.sh` for applications and `src/codec_metrics/run.sh` for objective metrics).

## Usage
### Application
The script `run.sh` can be leveraged to evaluate four applications, emotion recogintion (ER), Automatic speaker verification (ASV), Automatic speech recognition (ASR) and Audio event classification (AEC)
```
bash run.sh
```

### Objective metrics
The script `src/codec_metrics/run.sh` can be leveraged to evaluate objective metrics
```shell
cd src/codec_metrics
bash run.sh speech librispeech
bash run.sh audio esc50
```

## Reference
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- [TaoRuijie' ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [Whisper](https://github.com/openai/whisper)
- [CLAP](https://github.com/microsoft/CLAP.git)