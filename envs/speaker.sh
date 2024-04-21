git clone https://github.com/hbwu-ntu/ECAPA-TDNN.git
conda create -n ECAPA python=3.7.9
source activate ECAPA
mv ECAPA-TDNN src/ASV
cd src/ASV
pip install -r requirements.txt