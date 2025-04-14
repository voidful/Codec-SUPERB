source ~/.bashrc
conda activate llm
python trainer.py --task asr --dataset voidful/librispeech_codec --save_total_limit 20 --output_dir academicodec_hifi_16k_320d_large_uni --codec_setup academicodec_hifi_16k_320d_large_uni_unit
