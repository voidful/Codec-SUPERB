## 1.1 Application - Emotion
- conda create -n emo2vec python=3.8; conda activate emo2vec
- pip install fairseq
- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
- pip install -U funasr modelscope

## 1.2 Application - Speaker verification
### Environment installment
- git clone https://github.com/TaoRuijie/ECAPA-TDNN.git
- conda create -n ECAPA python=3.7.9; conda activate ECAPA
- mv ECAPA-TDNN src/ASV;
- cd src/ASV
- pip install -r requirements.txt
- comment #46 and #47 in src/ASV/trainECAPAModel.py (very important)

### Data prepare
- wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
- wget wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
- mv veri_test2.txt src/ASV
