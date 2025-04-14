source ~/.bashrc
conda activate llm
python trainer.py --task asr --dataset voidful/librispeech_codec --save_total_limit 20 --output_dir encodec_24k_24bps --codec_setup encodec_24k_24bps_unit