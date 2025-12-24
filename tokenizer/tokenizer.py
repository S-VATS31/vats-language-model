from configs.model_args.args_5M import ModelArgs
from transformers import PreTrainedTokenizerBase, AutoTokenizer

def get_tokenizer(model_args: ModelArgs) -> PreTrainedTokenizerBase:
    """Get HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_args.pad_token_id = tokenizer.pad_token_id
    model_args.eos_token_id = tokenizer.eos_token_id
    model_args.vocab_size = tokenizer.vocab_size
    return tokenizer
