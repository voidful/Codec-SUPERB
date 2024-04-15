FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
WORKDIR /workspace

RUN apt-get update && apt-get install -y wget git unzip

RUN git clone -b SLT_Challenge https://github.com/voidful/Codec-SUPERB.git /workspace/Codec-SUPERB

RUN conda create -y -n emo2vec python=3.8
SHELL ["conda", "run", "-n", "emo2vec", "/bin/bash", "-c"]
RUN pip install fairseq
RUN pip install funasr
RUN pip install modelscope
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

#RUN git clone https://github.com/hbwu-ntu/ECAPA-TDNN.git /workspace/ECAPA-TDNN
#RUN conda create -y -n ECAPA python=3.8
#SHELL ["conda", "run", "-n", "ECAPA", "/bin/bash", "-c"]
#WORKDIR /workspace/Codec-SUPERB
#RUN mkdir -p src/ASV && mv ../ECAPA-TDNN src/ASV
#RUN pip install -r requirements.txt
#RUN wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
#RUN wget https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip
#RUN unzip vox1_test_wav.zip
#RUN mv veri_test2.txt src/ASV

RUN conda create -y -n whisper python=3.8
SHELL ["conda", "run", "-n", "whisper", "/bin/bash", "-c"]
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
    pip install librosa==0.10.1 pandas==2.0.3 && \
    pip install -U openai-whisper && \
    pip install jiwer==3.0.3 editdistance==0.8.1


