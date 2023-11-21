from datasets import load_dataset, DatasetDict
from jiwer import wer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from pathlib import Path
import os
from codec_bart_model import BartCodecForConditionalGeneration
import random
import argparse
os.environ["WANDB_DISABLED"] = "true"
parser = argparse.ArgumentParser(description='Run audio encoding-decoding experiments.')
parser.add_argument('--codec', type=str, required=True,
                    help='Name of the codec to train ASR')
parser.add_argument('--epoch', type=int, default=20,
                    help='Epoch to train ASR')
args = parser.parse_args()
random.seed(100)
# Load dataset and tokenizer
# datasets_dict = DatasetDict()
# dataset_dict_path = Path('/mnt/data/stan/AudioDecBenchmark/cache/AudioDecBenchmark___librispeech_asr')
# if dataset_dict_path.exists():
#     datasets_dict = datasets_dict.load_from_disk(dataset_dict_path=dataset_dict_path)

# dataset = datasets_dict[args.codec]

train_dataset = load_dataset("AudioDecBenchmark/librispeech_asr", cache_dir="cache")
valid_dataset = load_dataset("AudioDecBenchmark/librispeech_asr_test", cache_dir="cache")
train_dataset = train_dataset[args.codec]
valid_dataset = valid_dataset[args.codec]
codec_book_num = len(train_dataset[0]["unit"])
num_proc=8

tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit", cache_dir="cache")
model = BartCodecForConditionalGeneration.from_pretrained("voidful/bart-base-unit", cache_dir="cache")
model.model.encoder.set_learning_weight(codec_book_num=codec_book_num)

# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./checkpoint/{args.codec}",
    logging_dir=f"./logs/{args.codec}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=args.epoch,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    gradient_accumulation_steps=2,
    warmup_steps=0,
    logging_steps=50,
    save_steps=500,
    eval_steps=10,
    learning_rate=5e-5,
    weight_decay=0,
    predict_with_generate=True,
    generation_max_length=300,
    fp16=True,
    save_total_limit=1,
)

# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# Define training and validation functions
# def process_data_to_model_inputs(batch):
#     max_len = 1023
#     labels = tokenizer(batch["text"], padding=True, truncation=True, max_length=max_len).input_ids
#     # Replace pad_token_id (0) with -100
#     labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
#     batch["labels"] = labels

#     input_datas = []
#     for b in range(len(batch['text'])):
#         encodec_input = []
#         for i in range(len(batch['unit'][0])):
#             encodec_input.append(
#                 tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1024}" for u in batch[f'unit'][b][i]]))
#         input_datas.append(encodec_input)
#     # Pad the input data sequences and create attention masks
#     padded_input_datas = []
#     attention_masks = []
#     for input_data in input_datas:
#         padded_input_data = []
#         for seq in input_data:
#             seq_len = len(seq)
#             padded_seq = seq + [tokenizer.pad_token_id] * (max_len - seq_len)
#             mask = [1] * seq_len + [0] * (max_len - seq_len)
#             padded_input_data.extend(padded_seq)
#         padded_input_datas.append(padded_input_data)
#         attention_masks.append(mask)
#     batch["input_ids"] = padded_input_datas
#     batch["feature_type_ids"] = [[1] * max_len * len(batch['unit'][0])] * len(batch["input_ids"])
#     batch["attention_mask"] = attention_masks
#     return batch

def process_data_to_model_inputs(batch):
    max_len = 1023
    labels = tokenizer(batch["text"], padding=True, truncation=True, max_length=max_len).input_ids
    # Replace pad_token_id (0) with -100
    labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
    batch["labels"] = labels

    input_datas = []
    for b in range(len(batch['text'])):
        encodec_input = []
        for i in range(len(batch['unit'][0])):
            encodec_input.append(
                tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1024}" for u in batch[f'unit'][b][i]]))
        input_datas.append(encodec_input)
    # Pad the input data sequences and create attention masks
    padded_input_datas = []
    attention_masks = []
    for input_data in input_datas:
        padded_input_data = []
        for seq in input_data:
            seq_len = len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * (max_len - seq_len)
            mask = [1] * seq_len + [0] * (max_len - seq_len)
            padded_input_data.append(padded_seq)
        padded_input_datas.append(padded_input_data)
        attention_masks.append(mask)
    batch["input_ids"] = padded_input_datas
    # batch["feature_type_ids"] = [[1] * max_len ] * len(batch["input_ids"])
    batch["attention_mask"] = attention_masks
    return batch

def filter_examples(example):
    return len(example[f"unit"][0]) <= 1000


train_dataset = train_dataset.filter(filter_examples, num_proc=num_proc,)
valid_dataset = valid_dataset.filter(filter_examples, num_proc=num_proc,)

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=training_args.per_device_train_batch_size,
    num_proc=num_proc,
)
valid_dataset = valid_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=training_args.per_device_eval_batch_size,
    num_proc=num_proc,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    predictions = [i[i != -100] for i in predictions]
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
trainer.train(resume_from_checkpoint=os.path.exists(f"./checkpoint/{args.codec}"),)
# trainer.evaluate()
