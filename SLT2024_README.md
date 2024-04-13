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
