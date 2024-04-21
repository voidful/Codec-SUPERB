conda create -n codec_metric python=3.8
source activate codec_metric
pip install -r requirements.txt
pip install torch==1.13.1 torchaudio==1.13.1
pip install torchvision
pip install numpy
pip install pandaspip install scikit-learn
pip install tqdm
pip install transformers
pip install torchlibrosa
pip install mir_eval
pip install soundfile
pip install librosa
pip install google
cd src/codec_metrics
wget -nc https://huggingface.co/Dongchao/pre_trained_model/resolve/main/visqol.zip
unzip -n visqol.zip