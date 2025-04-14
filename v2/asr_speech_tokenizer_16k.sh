source ~/.bashrc
conda activate llm
python trainer.py --task asr --dataset voidful/librispeech_codec --save_total_limit 20 --output_dir speech_tokenizer_16k --codec_setup speech_tokenizer_16k_unit