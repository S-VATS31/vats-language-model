from src.generation import AutoregressiveTokenGenerator
from configs.model_args.args_5M import ModelArgs
from transformers import PreTrainedTokenizerBase, AutoTokenizer

def _get_tokenizer(model_args: ModelArgs) -> PreTrainedTokenizerBase:
    """Get tokenizer for token generation."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_args.pad_token_id = tokenizer.pad_token_id
    model_args.eos_token_id = tokenizer.eos_token_id
    model_args.vocab_size = tokenizer.vocab_size
    return tokenizer

def generate(prompt: str) -> str:
    """High level generation API."""
    model_args = ModelArgs()
    tokenizer = _get_tokenizer(model_args)
    generator = AutoregressiveTokenGenerator(model_args, tokenizer)
    return generator.generate_tokens(prompt)

print(generate("hello world"))