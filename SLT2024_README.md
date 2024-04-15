## 1.1 Application - Emotion
### Environment installment
- conda create -n emo2vec python=3.8; conda activate emo2vec
- pip install fairseq
- pip install -U funasr modelscope
- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

### Data prepare
- Haibin will prepare the data (RAVDESS)

## 1.2 Application - Speaker verification
### Environment installment
- git clone https://github.com/hbwu-ntu/ECAPA-TDNN.git
- conda create -n ECAPA python=3.7.9; conda activate ECAPA
- mv ECAPA-TDNN src/ASV;
- cd src/ASV
- pip install -r requirements.txt

### Data prepare
- wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
- wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
- mv veri_test2.txt src/ASV

## 1.3 Application - Automatic speech recognition 
### Environment installment
- conda create -n whisper python=3.8; conda activate whisper
- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
- pip install librosa==0.10.1
- pip install pandas==2.0.3
- pip install -U openai-whisper
- pip install jiwer==3.0.3
- pip install editdistance==0.8.1

### Data Structure Guidelines
- **Directory Placement**: Place your synthesized data within the directory specified by `syn_path`.
- **File Naming**: Each synthesized data file should be named to match its corresponding original LibriSpeech data file, excluding the file extension.
- **Directory Structure**: The script is designed to recursively search through subfolders within `syn_path`. You may either replicate the directory structure of the original LibriSpeech dataset, or place all synthesized files directly in `syn_path` without subfolders.

## 1.4 Application - Audio Event Classification
### Environment installment
- git clone https://github.com/microsoft/CLAP.git
- conda create -n aec_clap python=3.8; conda activate aec_clap
- mv CLAP src/AEC
- cd src/AEC
- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
- pip install numpy==1.24.4
- pip install pandas==2.0.3
- pip install scikit-learn==1.3.2
- pip install tqdm==4.66.2
- pip install transformers==4.39.3
- pip install torchlibrosa

## Docker
build docker image
```shell
docker build -t slt .
```

run docker image
```shell
docker run -it slt /bin/bash
```
