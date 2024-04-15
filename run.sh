#!/bin/bash

stage=3

# For different stage, set different syn_path and ref_path
syn_path=/syn/path
ref_path=/ref/path
outdir=exps
result_log=logs
mkdir -p exps

echo "Codec SUPERB application evaluation" | tee ${result_log}
if [ $stage -le 1 ]; then

    echo -e "\nStage 1: Run speech emotion recognition." | tee -a $result_log
    source ~/.bashrc
    conda activate emo2vec
    model_type='iic/emotion2vec_base_finetuned'

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/SER/evaluation.py \
            --syn_path ${syn_path} \
            --ref_path ${ref_path} \
            --model_type $model_type \
            2>&1 | tee $outdir/emo.log
    fi

    if [ "do" ]; then
        Acc_ref_audio=$(grep -oP 'Acc_ref_audio \K\d+\.\d+%' $outdir/emo.log)
        echo Acc_ref_audio: $Acc_ref_audio | tee -a $result_log
        Acc_ground_truth=$(grep -oP 'Acc_ground_truth \K\d+\.\d+%' $outdir/emo.log)
        echo Acc_ground_truth: $Acc_ground_truth | tee -a $result_log
    fi

fi

if [ $stage -le 2 ]; then

    echo -e "\nStage 2: Run speaker related evaluation." | tee -a $result_log
    source ~/.bashrc
    conda activate ECAPA

    if [ ! -f "src/ASV/resyn_trial.txt" ]; then

        echo -e "Parsing the resyn_trial.txt for resyn wavs"  | tee -a $result_log
        while IFS= read -r line; do
            IFS=' ' read -r -a array <<< "$line"
            array[1]="$syn_path/${array[1]}"
            array[2]="$syn_path/${array[2]}"
            echo "${array[@]}" >> src/ASV/resyn_trial.txt
        done < src/ASV/veri_test2.txt

    fi

    echo -e "\nRun speaker verification." | tee -a $result_log

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/ASV/trainECAPAModel.py \
            --eval \
            --initial_model src/ASV/exps/pretrain.model \
            --eval_list src/ASV/resyn_trial.txt \
            2>&1 | tee $outdir/asv.log
    fi

    if [ "do" ]; then
        eer=$(grep 'EER' $outdir/asv.log | sed -n 's/.*EER \([^,]*\),.*/\1/p')
        echo EER: $eer | tee -a $result_log
    fi

    if [ ! -f "src/ASV/sim_trial.txt" ]; then
        echo -e "\nGenerate similarity trials." | tee -a $result_log
        awk -v syn_path="$syn_path" -v ref_path="$ref_path" \
                '{print syn_path "/" $2, ref_path "/" $2}' src/ASV/veri_test2.txt | \
                sort -u > src/ASV/sim_trial.txt
    fi

    echo -e "\nRun speaker similarity evaluation." | tee -a $result_log

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/ASV/trainECAPAModel.py \
            --eval_sim \
            --initial_model src/ASV/exps/pretrain.model \
            --sim_eval_list src/ASV/sim_trial.txt \
            2>&1 | tee $outdir/asv_sim.log
    fi

    if [ "do" ]; then
        spk_sim=$(grep -oP 'Similarity \K\d+\.\d+%' $outdir/asv_sim.log)
        echo Speaker similarity: $spk_sim | tee -a $result_log
    fi

fi

if [ $stage -le 3 ]; then

    echo -e "\nStage 3: Run automatic speech recognition." | tee -a $result_log
    source ~/.bashrc
    conda activate whisper

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/ASR/evaluation.py \
            --syn_path ${syn_path} \
            2>&1 | tee $outdir/asr.log
    fi

    if [ "do" ]; then
        ref_wer=$(grep -oP 'Ref WER: \K\d+\.\d+%' $outdir/asr.log)
        echo Ref WER: $ref_wer | tee -a $result_log
        syn_wer=$(grep -oP 'Syn WER: \K\d+\.\d+%' $outdir/asr.log)
        echo Syn WER: $syn_wer | tee -a $result_log
        ref_ed=$(grep -oP 'Ref Edit Distance: \K\d+\.\d+' $outdir/asr.log)
        echo Ref Edit Distance: $ref_ed | tee -a $result_log
        syn_ed=$(grep -oP 'Syn Edit Distance: \K\d+\.\d+' $outdir/asr.log)
        echo Syn Edit Distance: $syn_ed | tee -a $result_log
    fi

fi

if [ $stage -le 4 ]; then

    echo -e "\nStage 4: Run audio event classification." | tee -a $result_log
    source ~/.bashrc
    conda activate aec_clap

    if [ "do" ]; then
        CUDA_VISIBLE_DEVICES=0 \
        python src/AEC/evaluation.py \
            --syn_path ${syn_path} \
            2>&1 | tee $outdir/aec.log
    fi

    if [ "do" ]; then
        ref_acc=$(grep -oP 'Acc_ground_truth: \K\d+\.\d+%' $outdir/aec.log)
        echo Ref ACC: $ref_acc | tee -a $result_log
        syn_acc=$(grep -oP 'Acc_resync_audio: \K\d+\.\d+%' $outdir/aec.log)
        echo Syn ACC: $syn_acc | tee -a $result_log
        syn_cos=$(grep -oP 'Cos_similarity: \K\d+\.\d+%' $outdir/aec.log)
        echo Syn COS: $syn_cos | tee -a $result_log
    fi

fi
