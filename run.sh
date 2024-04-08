#!/bin/bash

stage=1

syn_path=/syn/path
ref_path=/ref/path
outdir=exps
result_log=logs
mkdir -p exps

echo "Codec SUPERB application evaluation" > ${result_log}
if [ $stage -eq 1 ]; then

    echo -e "\nStage 1: Run speech emotion recognition." | tee -a $result_log
    eval "$(conda shell.bash hook)"
    conda activate emo2vec
    model_type='iic/emotion2vec_base_finetuned'

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=1 \
        python src/SER/evaluation.py \
            --syn_path ${syn_path} \
            --ref_path ${ref_path} \
            --model_type $model_type \
            2>&1 | tee $outdir/emo_sim.log
    fi

    if [ "do" ]; then
        cat $outdir/emo_sim.log \
            | perl -e '$sum = 0; while(<>){if(/Accuracy:\s+(\S+)$/){$sum +=$1; $count++;}}; print "Accuracy: ", $sum/$count, "\n";' | tee -a $result_log
    fi

fi