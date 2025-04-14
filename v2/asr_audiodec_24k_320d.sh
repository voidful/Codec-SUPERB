source ~/.bashrc
conda activate llm
python trainer.py --task asr --dataset voidful/librispeech_codec --save_total_limit 20 --output_dir audiodec_24k_320d --codec_setup audiodec_24k_320d_unit
