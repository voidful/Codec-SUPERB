# Codec-SUPERB @ SLT 2024: Sound Codec Speech Processing Universal Performance Benchmark

Codec-SUPERB challenge @ SLT 2024, aims to enable a fair comparison of all current codec models and stimulate the development of more advanced codec models. 

<a href='https://codecsuperb.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2402.13071'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## 1. Environment installation
```shell
conda create -n codec-superb python=3.8
conda activate codec-superb
pip install --no-dependencies -r requirements.txt
git clone https://github.com/microsoft/CLAP.git src/AEC/CLAP
git clone https://github.com/hbwu-ntu/ECAPA-TDNN.git src/ASV
wget -P src/codec_metrics -nc https://huggingface.co/Dongchao/pre_trained_model/resolve/main/visqol.zip
unzip -n src/codec_metrics/visqol.zip -d src/codec_metrics
```

## 2. Data download
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

## 3. Usage
### 3.1 Application
The script `run.sh` can be leveraged to evaluate four applications, emotion recogintion (ER), Automatic speaker verification (ASV), Automatic speech recognition (ASR) and Audio event classification (AEC). You should change `syn_path` and `ref_path` in `run.sh`.
```
bash run.sh
```

### 3.2 Objective metrics
The script `src/codec_metrics/run.sh` can be leveraged to evaluate objective metrics. You should change `syn_path` and `ref_path` in `src/codec_metrics/run.sh`.
```shell
cd src/codec_metrics
bash run.sh speech librispeech
bash run.sh audio esc50
```

## 4. Results submission
- Open an issue.
- Copy results in `exps/results.txt` and `src/codec_metrics/exps/results.txt`, and paste them in the issue.

## Citation
If you find this codebase useful, please cite our work as:
```Tex
@misc{wu2024codecsuperb,
      title={Codec-SUPERB: An In-Depth Analysis of Sound Codec Models}, 
      author={Haibin Wu and Ho-Lam Chung and Yi-Cheng Lin and Yuan-Kuei Wu and Xuanjun Chen and Yu-Chi Pai and Hsiu-Hsuan Wang and Kai-Wei Chang and Alexander H. Liu and Hung-yi Lee},
      year={2024},
      eprint={2402.13071},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## References：
- [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- [TaoRuijie' ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- [Whisper](https://github.com/openai/whisper)
- [CLAP](https://github.com/microsoft/CLAP.git)
