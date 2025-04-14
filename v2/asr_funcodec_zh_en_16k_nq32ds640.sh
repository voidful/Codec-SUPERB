source ~/.bashrc
conda activate llm
python trainer.py --task asr --dataset voidful/librispeech_codec --save_total_limit 20 --output_dir funcodec_zh_en_16k_nq32ds640 --codec_setup funcodec_zh_en_16k_nq32ds640_unit