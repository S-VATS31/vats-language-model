from src.generation import AutoregressiveTokenGenerator
from configs.model_args.args_5M import ModelArgs
from tokenizer.tokenizer import get_tokenizer

def generate(prompt: str) -> str:
    """High level generation API."""
    model_args = ModelArgs()
    tokenizer = get_tokenizer(model_args)
    generator = AutoregressiveTokenGenerator(model_args, tokenizer)
    return generator.generate_tokens(prompt)

print(generate("hello world"))
