from datasets import load_dataset
from jiwer import wer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from downstream.codec_bart_model import BartCodecForConditionalGeneration

dataset = load_dataset("voidful/librispeech_encodec")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
model = BartCodecForConditionalGeneration.from_pretrained("voidful/bart-base-unit")
train_dataset = dataset['trainclean100']
valid_dataset = dataset['validationclean']
training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    logging_dir="./logs",
    evaluation_strategy="epoch",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=0,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    learning_rate=5e-5,
    weight_decay=0,
    predict_with_generate=True,
    generation_max_length=300,
    fp16=True,
    save_total_limit=3,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
max_sequence_len = 1023
codec_layer = 8


def process_data_to_model_inputs(batch):
    labels = tokenizer(batch["text"], padding=True, truncation=True, max_length=max_sequence_len).input_ids
    labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
    batch["labels"] = labels

    input_datas = []
    for b in range(len(batch['text'])):
        encodec_input = []
        for i in range(codec_layer):
            encodec_input.append(
                tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1024}" for u in batch[f'encodec_{i}'][b]]))
        input_datas.append(encodec_input)
    # Pad the input data sequences and create attention masks
    padded_input_datas = []
    attention_masks = []
    for input_data in input_datas:
        padded_input_data = []
        for seq in input_data:
            seq_len = len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * (max_sequence_len * codec_layer - seq_len)
            mask = [1] * seq_len + [0] * (max_sequence_len * codec_layer - seq_len)
            padded_input_data.extend(padded_seq)
        padded_input_datas.append(padded_input_data)
        attention_masks.append(mask)
    batch["input_ids"] = padded_input_datas
    batch["attention_mask"] = attention_masks
    return batch


def filter_examples(example):
    return len(example[f"encodec_0"]) <= 1000


train_dataset = train_dataset.filter(filter_examples)
valid_dataset = valid_dataset.filter(filter_examples)

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=training_args.per_device_train_batch_size
)
valid_dataset = valid_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=training_args.per_device_eval_batch_size
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_value = wer(decoded_labels, decoded_preds)
    print("pred_result")
    print("=================================")
    for i in range(10):
        print(decoded_labels[i], " ///// ", decoded_preds[i])
    print("=================================")
    return {"wer": wer_value}


# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
# trainer.evaluate()
