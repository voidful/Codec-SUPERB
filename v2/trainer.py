import argparse
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


# Set training arguments
def set_training_args(args):
    return Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        evaluation_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=args.fp16,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=False if args.task == 'asr' else True
    )


# Main function to handle the training process
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Seq2Seq Training Script")

    # Arguments related to dataset and model
    parser.add_argument("--output_dir", type=str, default="./training_output", help="Directory to save model output.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to save logs.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps for learning rate.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps during training.")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps interval for saving the model.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps interval for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimization.")
    parser.add_argument("--generation_max_length", type=int, default=300,
                        help="Maximum length for sequence generation.")
    parser.add_argument("--fp16", action="store_true", help="Whether to use FP16 precision.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total number of checkpoints to keep.")
    parser.add_argument("--codec_setup", type=str, default="encodec_24k_3bps_unit")
    parser.add_argument("--dataset", type=str, default="voidful/librispeech_codec")
    parser.add_argument("--model", type=str, default="voidful/bart-base-codec")
    parser.add_argument("--task", type=str, choices=["asr", "tts", "qa"], default="asr")

    args = parser.parse_args()

    if args.task == "asr":
        from task.asr import (load_data_and_model, split_and_concatenate_dataset, filter_examples,
                              process_data_to_model_inputs, compute_metrics)
    elif args.task == "tts":
        from task.tts import (load_data_and_model, split_and_concatenate_dataset, filter_examples,
                              process_data_to_model_inputs, compute_metrics)
    elif args.task == "qa":
        from task.qa import (load_data_and_model, split_and_concatenate_dataset, filter_examples,
                             process_data_to_model_inputs, compute_metrics)
    # Load data and model
    dataset, tokenizer, model = load_data_and_model(args.codec_setup,
                                                    args.dataset,
                                                    args.model)
    dataset.cleanup_cache_files()

    # Split and concatenate dataset
    train_dataset, valid_dataset = split_and_concatenate_dataset(dataset)

    # Filter datasets
    # train_dataset = train_dataset.filter(filter_examples)
    # valid_dataset = valid_dataset.filter(filter_examples)

    # Set training arguments using the parsed arguments
    training_args = set_training_args(args)

    # Define a data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Process datasets
    train_dataset = train_dataset.map(
        lambda batch: process_data_to_model_inputs(batch, tokenizer, args.codec_setup),
        batched=True,
        remove_columns=train_dataset.column_names,
        batch_size=training_args.per_device_train_batch_size
    )

    valid_dataset = valid_dataset.map(
        lambda batch: process_data_to_model_inputs(batch, tokenizer, args.codec_setup),
        batched=True,
        remove_columns=valid_dataset.column_names,
        batch_size=training_args.per_device_eval_batch_size
    )

    # Create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
