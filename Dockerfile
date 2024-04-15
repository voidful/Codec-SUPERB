FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    wget \
    git \
    gcc \
    libpq-dev \
    unzip

RUN git clone -b SLT_Challenge https://github.com/voidful/Codec-SUPERB.git /workspace/Codec-SUPERB

# Application - Emotion
RUN conda create -y -n a_emo python=3.8
SHELL ["conda", "run", "-n", "a_emo", "/bin/bash", "-c"]
RUN pip install fairseq
RUN pip install funasr
RUN pip install modelscope
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Application - Speaker verification
RUN git clone https://github.com/hbwu-ntu/ECAPA-TDNN.git /workspace/ECAPA-TDNN
RUN conda create -y -n a_sv python=3.8
SHELL ["conda", "run", "-n", "a_sv", "/bin/bash", "-c"]
WORKDIR /workspace/Codec-SUPERB
RUN mv ../ECAPA-TDNN src/ASV
RUN pip install -r src/ASV/requirements.txt

# Application - Automatic speech recognition
RUN conda create -y -n a_asr python=3.8
SHELL ["conda", "run", "-n", "a_asr", "/bin/bash", "-c"]
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
    pip install librosa==0.10.1 pandas==2.0.3 && \
    pip install -U openai-whisper && \
    pip install jiwer==3.0.3 editdistance==0.8.1

# Application - Audio Event Classification
RUN git clone https://github.com/microsoft/CLAP.git /workspace/CLAP
RUN conda create -y -n a_aec python=3.8
SHELL ["conda", "run", "-n", "a_aec", "/bin/bash", "-c"]
WORKDIR /workspace/Codec-SUPERB
RUN mv ../CLAP src/AEC
WORKDIR /workspace/Codec-SUPERB/src/AEC
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN pip install numpy==1.24.4
RUN pip install pandas==2.0.3
RUN pip install scikit-learn==1.3.2
RUN pip install tqdm==4.66.2
RUN pip install transformers==4.39.3
RUN pip install torchlibrosa

# Signal
WORKDIR /workspace/Codec-SUPERB
RUN pip install -r requirements.txt
# Install VisQOL
# Install Bazel for building ViSQOL
RUN apt-get install -y apt-transport-https curl gnupg
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
RUN mv bazel-archive-keyring.gpg /usr/share/keyrings
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y bazel-5.3.2
RUN git clone https://github.com/google/visqol.git /workspace/visqol
WORKDIR /workspace/visqol
RUN /usr/bin/bazel-5.3.2 build :visqol -c opt
RUN pip install .
#
WORKDIR /workspace/Codec-SUPERB