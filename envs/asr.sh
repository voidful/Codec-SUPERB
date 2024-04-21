conda create -n whisper python=3.8
source activate whisper
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install librosa==0.10.1
pip install pandas==2.0.3
pip install -U openai-whisper
pip install jiwer==3.0.3
pip install editdistance==0.8.1