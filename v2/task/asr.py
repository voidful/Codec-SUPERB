from datasets import load_dataset, concatenate_datasets
from jiwer import wer
from transformers import (
    AutoTokenizer
)
from codec_bart_model import BartCodecForConditionalGeneration


# Split and concatenate multiple training datasets
def split_and_concatenate_dataset(dataset):
    train_ds = concatenate_datasets([
        dataset['train.clean.100'],
        dataset['train.clean.360'],
        dataset['train.other.500'],
    ])
    valid_ds = concatenate_datasets([
        dataset['validation.clean'],
        dataset['validation.other']
    ])
    return train_ds, valid_ds


# Load dataset and tokenizer
def load_data_and_model(codec_name, dataset_config="voidful/librispeech_codec", model_config="voidful/bart-base-codec"):
    dataset = load_dataset(dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    model = BartCodecForConditionalGeneration.from_pretrained(model_config)
    data_item = next(iter(dataset.values()))
    model.set_num_codec_layers(len(data_item[codec_name][0]))
    return dataset, tokenizer, model


# Process data for model inputs
def process_data_to_model_inputs(batch, tokenizer, codec_name, max_len=1023):
    labels = tokenizer(batch["text"], padding=True, truncation=True, max_length=max_len).input_ids
    labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
    batch["labels"] = labels

    input_datas = [
        [
            tokenizer.convert_tokens_to_ids([f"v_tok_{i + n_l * 1024}" for i in l])
            for n_l, l in enumerate(b)
        ]
        for b in batch[codec_name]
    ]
    # Pad the input data sequences and create attention masks
    padded_input_datas = []
    attention_masks = []
    for input_data in input_datas:
        padded_input_data = []
        mask_id = []
        for seq in input_data:
            seq_len = len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * (max_len - seq_len)
            mask_id = [1] * seq_len + [0] * (max_len - seq_len)
            padded_input_data.append(padded_seq)
        padded_input_datas.append(padded_input_data)
        attention_masks.append(mask_id)
    batch["input_ids"] = padded_input_datas
    batch["attention_mask"] = attention_masks
    return batch


# Filter examples based on sequence length
def filter_examples(example):
    return True


# Compute metrics (WER)
def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    predictions = [i[i != -100] for i in predictions]
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    wer_value = wer(decoded_labels, decoded_preds)

    print("pred_result")
    print("=================================")
    for i in range(10):
        print(decoded_labels[i], " ///// ", decoded_preds[i])
    print("=================================")

    return {"wer": wer_value}
