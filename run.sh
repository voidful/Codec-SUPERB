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

if [ $stage -eq 2 ]; then

    echo -e "\nStage 2: Run speaker verification." | tee -a $result_log
    eval "$(conda shell.bash hook)"
    conda activate ECAPA

    if [ "do" ]; then

        echo "Parsing the trial.txt for resyn wavs"
        while IFS= read -r line; do
            IFS=' ' read -r -a array <<< "$line"
            array[1]="$syn_path/${array[1]}"
            array[2]="$syn_path/${array[2]}"
            echo "${array[@]}" >> src/ASV/resyn_trial.txt
        done < src/ASV/veri_test2.txt
    fi

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/ASV/trainECAPAModel.py \
            --eval \
            --initial_model src/ASV/exps/pretrain.model \
            --eval_list src/ASV/resyn_trial.txt \
            2>&1 | tee $result_log
    fi

fi