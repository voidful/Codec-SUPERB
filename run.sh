#!/bin/bash

stage=4

syn_path=/syn/path
ref_path=/ref/path
outdir=exps
result_log=$outdir/results.txt
mkdir -p ${outdir}/logs

if [ ! -f ${result_log} ]; then
    echo "Codec SUPERB application evaluation" | tee ${result_log}
fi

if [ $stage -eq 1 ]; then

    echo -e "\nStage 1: Run speech emotion recognition." | tee -a $result_log
    model_type='iic/emotion2vec_base_finetuned'

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/SER/evaluation.py \
            --syn_path ${syn_path}/RAVDESS/ravdess \
            --ref_path ${ref_path}/RAVDESS/ravdess \
            --model_type $model_type \
            2>&1 | tee ${outdir}/logs/emo.log
    fi

    if [ "do" ]; then
        Acc_ground_truth=$(grep -oP 'Acc_ground_truth \K\d+\.\d+%' ${outdir}/logs/emo.log)
        echo Acc: $Acc_ground_truth | tee -a $result_log
    fi

fi

if [ $stage -eq 2 ]; then

    echo -e "\nStage 2: Run speaker related evaluation." | tee -a $result_log
    if [ ! -f "src/ASV/veri_test2.txt" ]; then
        wget -P src/ASV https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
    fi

    # Check if resyn_trial.txt exists
    if [ -f "src/ASV/resyn_trial.txt" ]; then
        # If it exists, remove it
        rm "src/ASV/resyn_trial.txt"
    fi
    echo -e "Parsing the resyn_trial.txt for resyn wavs" | tee -a "$result_log"
    while IFS= read -r line; do
        IFS=' ' read -r -a array <<< "$line"
        array[1]="$syn_path/vox1_test_wav/wav/${array[1]}"
        array[2]="$syn_path/vox1_test_wav/wav/${array[2]}"
        echo "${array[@]}" >> "src/ASV/resyn_trial.txt"
    done < "src/ASV/veri_test2.txt"

    echo -e "\nRun speaker verification." | tee -a $result_log

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/ASV/trainECAPAModel.py \
            --eval \
            --initial_model src/ASV/exps/pretrain.model \
            --eval_list src/ASV/resyn_trial.txt \
            2>&1 | tee ${outdir}/logs/asv.log
    fi

    if [ "do" ]; then
        eer=$(grep 'EER' ${outdir}/logs/asv.log | sed -n 's/.*EER \([^,]*\),.*/\1/p')
        echo EER: $eer | tee -a $result_log
    fi

fi

if [ $stage -eq 3 ]; then

    echo -e "\nStage 3: Run automatic speech recognition." | tee -a $result_log

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/ASR/evaluation.py \
            --syn_path ${syn_path}/LibriSpeech \
            --ref_path ${ref_path} \
            2>&1 | tee ${outdir}/logs/asr.log
    fi

    if [ "do" ]; then
        syn_wer=$(grep -oP 'Syn WER: \K\d+\.\d+%' ${outdir}/logs/asr.log)
        echo WER: $syn_wer | tee -a $result_log
    fi

fi

if [ $stage -eq 4 ]; then

    echo -e "\nStage 4: Run audio event classification." | tee -a $result_log

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/AEC/evaluation.py \
            --syn_path ${syn_path}/ESC-50-master \
            --ref_path ${ref_path}/ESC-50-master \
            2>&1 | tee ${outdir}/logs/aec.log
    fi

    if [ "do" ]; then
        syn_acc=$(grep -oP 'Acc_resync_audio: \K\d+\.\d+%' ${outdir}/logs/aec.log)
        echo ACC: $syn_acc | tee -a $result_log
    fi

fi
