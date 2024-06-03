#!/bin/bash

speech_datasets=("crema_d" "fluent_speech_commands" "libri2Mix_test" "librispeech" "quesst" "snips_test_valid_subset" "vox_lingua_top10" "voxceleb1")
audio_datasets=("esc50" "fsd50k" "gunshot_triangulation")

for dataset in "${speech_datasets[@]}"; do
    bash run.sh speech $dataset
done

for dataset in "${audio_datasets[@]}"; do
    bash run.sh audio $dataset
done

if [ do ]; then

    result_log="exps/results.txt"
    echo "Log results" > $result_log
    echo "--------------------------------------------------" >> $result_log

    for log_file in exps/*.log; do

        filename=$(basename "$log_file")
        echo "File Name: $filename" >> $result_log
        cat "$log_file" >> $result_log
        echo "--------------------------------------------------" >> $result_log

    done

    if [ do ]; then
        python utils/log_overall_score.py --log_dir exps
    fi

fi