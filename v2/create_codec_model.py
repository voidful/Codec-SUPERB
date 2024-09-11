import copy
from transformers import AutoTokenizer, AutoModel


def load_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def add_custom_tokens(tokenizer, num_tokens):
    # Generate and add custom tokens to the tokenizer
    add_tokens = [f"v_tok_{u}" for u in range(num_tokens)]
    origin_vocab_size = tokenizer.vocab_size

    print("===ADD TOKEN===")
    num_added_toks = tokenizer.add_tokens(add_tokens)
    print(f"We have added {num_added_toks} tokens")
    print(f"Original vocab size: {origin_vocab_size}, New size: {len(tokenizer)}")

    return num_added_toks, origin_vocab_size


def reshape_model_embeddings(model, origin_vocab_size, num_added_toks):
    # Resize token embeddings to accommodate new tokens
    model.resize_token_embeddings(origin_vocab_size + num_added_toks)
    input_embedding = model.get_input_embeddings()

    # Copy the state dict weights for new tokens
    state_dict_weight = input_embedding.state_dict()['weight']
    state_dict_weight[origin_vocab_size:origin_vocab_size + num_added_toks] = copy.copy(
        state_dict_weight[100:100 + num_added_toks]
    )

    # Set the modified input embeddings back to the model
    model.set_input_embeddings(input_embedding)
    return state_dict_weight


def push_to_hub(tokenizer, model, repo_name):
    # Push tokenizer and model to Hugging Face Hub
    tokenizer.push_to_hub(repo_name)
    model.push_to_hub(repo_name)


def main():
    model_name = "facebook/bart-base"
    num_custom_tokens = 1024 * 9
    hub_repo_name = "voidful/bart-base-codec"

    # Load pre-trained tokenizer and model
    tokenizer, model = load_tokenizer_and_model(model_name)

    # Test tokenization before adding tokens
    print(tokenizer.tokenize("Hello world v_tok_10v_tok_1"))

    # Add custom tokens and reshape the model embeddings
    num_added_toks, origin_vocab_size = add_custom_tokens(tokenizer, num_custom_tokens)
    state_dict_weight = reshape_model_embeddings(model, origin_vocab_size, num_added_toks)

    print(f"New embedding shape: {state_dict_weight.shape}")

    # Test tokenization after adding tokens
    print(tokenizer.tokenize("Hello world v_tok_10v_tok_1"))

    # Push tokenizer and model to Hugging Face Hub
    push_to_hub(tokenizer, model, hub_repo_name)


if __name__ == "__main__":
    main()
