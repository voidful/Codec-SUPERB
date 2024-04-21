git clone https://github.com/microsoft/CLAP.git
conda create -n aec_clap python=3.8
source activate aec_clap
mv CLAP src/AEC
cd src/AEC
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install numpy==1.24.4
pip install pandas==2.0.3
pip install scikit-learn==1.3.2
pip install tqdm==4.66.2
pip install transformers==4.39.3
pip install torchlibrosa