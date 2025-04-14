source ~/.bashrc
conda activate llm
python trainer.py --task asr --dataset voidful/librispeech_codec --save_total_limit 20 --output_dir funcodec_en_libritts_16k_gr8nq32ds320 --codec_setup funcodec_en_libritts_16k_gr8nq32ds320_unit