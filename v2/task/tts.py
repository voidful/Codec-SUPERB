from datasets import load_dataset, concatenate_datasets
from jiwer import wer
from transformers import (
    AutoTokenizer
)
from codec_bart_model import BartCodecForConditionalGeneration


# Split and concatenate multiple training datasets
def split_and_concatenate_dataset(dataset):
    train_ds = concatenate_datasets([
        dataset['train'],
    ])
    valid_ds = concatenate_datasets([
        dataset['test'],
    ])
    return train_ds, valid_ds


# Load dataset and tokenizer
def load_data_and_model(codec_name, dataset_config="voidful/ljspeech_codec", model_config="voidful/bart-base-codec"):
    dataset = load_dataset(dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    model = BartCodecForConditionalGeneration.from_pretrained(model_config)
    data_item = next(iter(dataset.values()))
    model.set_num_codec_layers(1)
    return dataset, tokenizer, model


# Process data for model inputs
def process_data_to_model_inputs(batch, tokenizer, codec_name, max_len=1023):
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []
    for b in range(len(batch['text'])):
        data = tokenizer(batch["text"][b], padding='max_length', truncation=True, max_length=max_len)
        input_ids.append(data['input_ids'])
        attention_mask.append(data['attention_mask'])

        # first layer AR data
        encode_input = tokenizer.convert_tokens_to_ids([f"v_tok_{u}" for u in batch[codec_name][b][0]])
        decoder_input_id = [tokenizer.bos_token_id] + encode_input
        label = encode_input + [tokenizer.eos_token_id]
        decoder_input_ids.append(decoder_input_id)
        labels.append(label)

        # 1-7 layer NAR data
        for i in range(1, len(batch[codec_name][b])):
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (i - 1) * 1024}" for u in batch[codec_name][b][i - 1]])
            label = tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1024}" for u in batch[codec_name][b][i - 1]])
            input_ids.append(data['input_ids'])
            attention_mask.append(data['attention_mask'])
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)

    def pad_sequences_and_create_masks(sequences, max_length, padding_value):
        padded_sequences = [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]
        attention_masks = [[1 if token != padding_value else 0 for token in sequence] for sequence in padded_sequences]
        return padded_sequences, attention_masks

    # Pad decoder_input_ids and labels
    decoder_input_ids, decoder_attention_mask = pad_sequences_and_create_masks(decoder_input_ids, max_length=max_len,
                                                                               padding_value=tokenizer.pad_token_id)
    labels, _ = pad_sequences_and_create_masks(labels, max_length=max_len, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        "labels": labels
    }


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
    # Compute WER
    wer_value = wer([" ".join(filter(None, i.split("v_tok_"))) for i in decoded_labels],
                    [" ".join(filter(None, i.split("v_tok_"))) for i in decoded_preds])
    print("pred_result")
    print("=================================")
    for i in range(10):
        print("target:", labels[i])
        print("pred:", predictions[i])
        print("-----------------")
    print("=================================")
    return {"wer": wer_value}
