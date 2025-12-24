import torch
from torch.amp import autocast
from torch import LongTensor, Tensor    
from transformers import PreTrainedTokenizerBase

from configs.model_args.args_5M import ModelArgs
from inference.sampler import get_next_tokens
from src.model import CausalTransformer
from src.kv_cache import KVCache
from gpu_setup import device, dtype

class AutoregressiveTokenGenerator:
    """Module use for autoregressive token generation.
    
    Args:
        model_args (ModelArgs): Dataclass containing model hyperparameters.
        tokenizer (PreTrainedTokenizerBase): Tokenizer.
    """
    def __init__(
        self,
        model_args: ModelArgs,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model_args = model_args
        self.tokenizer = tokenizer

        self.model = CausalTransformer(model_args).to(device)
        self.model.eval()

        self.kv_cache = KVCache(
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            head_dim=model_args.d_model//model_args.num_heads,
            max_tokens=model_args.max_seq_len
        )

    def _generate_tokens(
        self,
        input_ids: LongTensor,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Generate tokens autoregressively using decoding methods.

        Args:
            input_ids (LongTensor): LongTensor containing tokenized ids.
            max_new_tokens (int): Maximum number of tokens the model can generate at a time.
            do_sample (bool): Whether to apply sampling or greedy decoding.
            temperature (float): Decoding method to encourage more randomness/determinism.
            top_k (int): Top-k logits to be sampled.
            top_p (float, optional): Top-p hyperparameter used as a threshold for masking out certain logits.
            pad_token_id (int, optional): Special value of the padding token to be masked out.
            eos_token_id (int, optional): End of sequence token appended to the end of each token.
            attention_mask (Tensor, optional): Padding mask of shape [B, T].
            use_cache (bool): Boolean to whether use the KV cache or not.

        Returns:
            torch.Tensor: Returns a tensor of generated tokens of shape [B, T].
        """
        B, T = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id)
        else:
            assert attention_mask.shape == input_ids.shape

        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        max_total_length = min(self.model_args.max_seq_len, T + max_new_tokens)
        actual_max_new_tokens = max_total_length - T
        if actual_max_new_tokens <= 0:
            return input_ids

        generated_ids = input_ids.clone() # [b, t]
        # all sequences start unfinished
        unfinished_sequences = torch.ones(B, dtype=torch.bool).to(device)

        with torch.no_grad():
            if use_cache:
                self.kv_cache.reset()
                self.kv_cache.initialize(B)

            logits = self.model(
                input_ids=generated_ids, padding_mask=attention_mask, use_cache=use_cache
            )
            
            for step in range(actual_max_new_tokens):
                current_seq_len = generated_ids.size(1)

                if current_seq_len >= self.model_args.max_seq_len:
                    break

                if not unfinished_sequences.any():
                    break

                if use_cache and step > 0 and current_seq_len < self.model_args.max_seq_len - 1:
                    if current_seq_len >= self.model_args.max_seq_len:
                        break 
                    # process last token for cached generation
                    # -1 takes the last token in the sequence
                    last_token = generated_ids[:, -1:].contiguous()
                    last_attention = torch.ones(B, 1, dtype=torch.bool).to(device)
                    last_attention = last_attention & unfinished_sequences[:, None]
                    logits = self.model(
                        input_ids=last_token, padding_mask=last_attention, use_cache=True
                    )
                else:
                    # process full sequence for non-cached generation
                    if attention_mask.size(1) < current_seq_len:
                        new_attention = torch.cat([attention_mask,
                            unfinished_sequences[:, None].expand(-1, current_seq_len - attention_mask.shape[1])
                        ], dim=1)
                    else:
                        new_attention = attention_mask[:, :current_seq_len]

                    logits = self.model(
                        input_ids=generated_ids, padding_mask=new_attention, use_cache=False
                    )

                # get logits for the last position
                next_token_logits = logits[:, -1, :] # [b, v]
                next_tokens = get_next_tokens(
                    logits=next_token_logits,
                    do_sample=do_sample,
                    temp=temperature,
                    top_k=top_k,
                    top_p=top_p
                ) # [b,]
                next_tokens = torch.where(unfinished_sequences, next_tokens, pad_token_id)
                generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=1)

                # update mask
                attention_mask = torch.cat([
                    attention_mask,
                    unfinished_sequences[:, None]
                ], dim=1)

                # check for eos token
                if eos_token_id is not None:
                    unfinished_sequences &= (next_tokens != eos_token_id)

        # reset cache
        if use_cache:
            self.kv_cache.reset()

        # clear gpu cache
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

        return generated_ids

    def _generate_from_prompt(
        self,
        prompt: str,
        attention_mask: Tensor | None = None
    ) -> str:
        """Generate tokens using a HuggingFace tokenizer.
        
        Args:
            prompt (str): Input string of text to be tokenized.
            attention_mask (Tensor, optional): Padding mask of shape [B, T].

        Returns:
            str: Generated text based on prompt.
        """
        if not prompt or not prompt.strip():
            return "Please enter a valid prompt."
        if self.model_args.max_new_tokens <= 0:
            return prompt
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            with autocast(device_type=device.type, dtype=dtype):
                generated_ids = self._generate_tokens(
                    input_ids=input_ids,
                    max_new_tokens=self.model_args.max_new_tokens,
                    temperature=self.model_args.temperature,
                    top_k=self.model_args.top_k,
                    top_p=self.model_args.top_p,
                    do_sample=self.model_args.do_sample,
                    pad_token_id=self.model_args.pad_token_id,
                    eos_token_id=self.model_args.eos_token_id,
                    attention_mask=attention_mask,
                    use_cache=self.model_args.use_cache
                )

        if self.model_args.return_only_new_tokens:
            return self.tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

        return self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

    def generate_tokens(self, prompt: str) -> str:
        """Public text generation API."""
        return self._generate_from_prompt(prompt)
