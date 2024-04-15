#!/bin/bash
cd scr/codec_metrics
wget https://huggingface.co/Dongchao/pre_trained_model/resolve/main/visqol.zip
unzip visqol.zip
stage=1
stop_stage=3
# For different stage, set different syn_path and ref_path
syn_path=/Users/dongchaoyang/Documents/emotion/rec
ref_path=/Users/dongchaoyang/Documents/emotion/gt
outdir=exps
result_log=logs
mkdir -p exps
conda activate codec_metric
echo "Codec SUPERB application evaluation" | tee ${result_log}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    echo -e "\nStage 1: Run SDR evaluation." | tee -a $result_log
    model_type=SDR
    python evaluation.py \
        --syn_path ${syn_path} \
        --ref_path ${ref_path} \
        --metric_name $model_type \
        2>&1 | tee $outdir/SDR.log

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo -e "\nStage 2: Run STFT_distance." | tee -a $result_log
    model_type=stft_dis
    python evaluation.py \
        --syn_path ${syn_path} \
        --ref_path ${ref_path} \
        --metric_name $model_type \
        2>&1 | tee $outdir/stft_distance.log

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    echo -e "\nStage 3: Run VISQOL." | tee -a $result_log
    model_type=visqol
    python evaluation.py \
        --syn_path ${syn_path} \
        --ref_path ${ref_path} \
        --metric_name $model_type \
        2>&1 | tee $outdir/VISQOL.log

fi
