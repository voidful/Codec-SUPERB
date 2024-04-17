#!/bin/bash
cd src/codec_metrics
wget -nc https://huggingface.co/Dongchao/pre_trained_model/resolve/main/visqol.zip
unzip -n visqol.zip
category=$1
# For different stage, set different syn_path and ref_path
syn_path=/home/ycevan/AudioDecBenchmark/sample_100/sample_100/audioset
ref_path=/home/ycevan/AudioDecBenchmark/sample_100/sample_100/audioset
outdir=exps
result_log=logs
mkdir -p exps
conda activate codec_metric
echo "Codec SUPERB application evaluation" | tee ${result_log}

if [ "$category" = "speech" ]; then
    stage=1
    stop_stage=6
    visqol_sr=16000
else
    stage=2
    stop_stage=4
    visqol_sr=48000
fi

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
        --target_sr $visqol_sr \
        2>&1 | tee $outdir/VISQOL.log

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    echo -e "\nStage 4: Run Mel Spectrogram Loss." | tee -a $result_log
    model_type=mel_loss
    python evaluation.py \
        --syn_path ${syn_path} \
        --ref_path ${ref_path} \
        --metric_name $model_type \
        2>&1 | tee $outdir/Mel_loss.log

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    echo -e "\nStage 5: Run STOI." | tee -a $result_log
    model_type=stoi
    python evaluation.py \
        --syn_path ${syn_path} \
        --ref_path ${ref_path} \
        --metric_name $model_type \
        2>&1 | tee $outdir/STOI.log

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

    echo -e "\nStage 6: Run PESQ." | tee -a $result_log
    model_type=pesq
    python evaluation.py \
        --syn_path ${syn_path} \
        --ref_path ${ref_path} \
        --metric_name $model_type \
        --target_sr 16000 \
        2>&1 | tee $outdir/PESQ.log

fi