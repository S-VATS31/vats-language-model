import torch
from torch import Tensor

from gpu_setup import device, dtype
from utils.logger import setup_logger

logger = setup_logger(__name__,  "inference.log")

class KVCache:
    """KV caching module."""
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        head_dim: int,
        max_tokens: int,
    ):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_tokens = max_tokens

        self.cache = None
        self.batch_size = None
        self.current_tokens = None

    def initialize(self, batch_size: int) -> None:
        """Initialize cache using given batch size."""
        self.batch_size = batch_size
        self.current_tokens = 0

        # initialize cache with zero tensors
        self.cache = [
            {
                "k": torch.ones((
                    self.batch_size, self.num_heads, self.max_tokens, self.head_dim
                ), device=device, dtype=dtype),
                "v": torch.ones((
                    self.batch_size, self.num_heads, self.max_tokens, self.head_dim
                ), device=device, dtype=dtype)
            }
            for _ in range(self.num_layers)
        ]

    def update(self, k: Tensor, v: Tensor, layer_idx: int) -> None:
        """Update cache using new KV's with respect to specific layer."""
        if self.cache is None or k.size(0) != self.batch_size:
            self.initialize(k.size(0))

        # get new tokens from new input tensor
        new_tokens = k.size(2)

        # check if cache has space
        if self.current_tokens + new_tokens > self.max_tokens:
            current_tokens_space = self.max_tokens - self.current_tokens
            if current_tokens_space <= 0:
                logger.info("No space in cache left, exiting.")
                return
            
            # truncate to current tokens space over seqlen dim
            k = k[:, :, :current_tokens_space]
            v = v[:, :, :current_tokens_space]
            logger.info(f"Truncated {new_tokens-current_tokens_space} tokens.")
            new_tokens = current_tokens_space

        # update cache over seqlen dim
        self.cache[layer_idx]["k"][:, :, self.current_tokens:self.current_tokens+new_tokens] = k
        self.cache[layer_idx]["v"][:, :, self.current_tokens:self.current_tokens+new_tokens] = v

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Get key and value tensors for all tokens in cache."""
        if self.cache is None:
            return None, None
        
        return (
            self.cache[layer_idx]["k"][:, :, :self.current_tokens],
            self.cache[layer_idx]["v"][:, :, :self.current_tokens],
        )

    def reset(self) -> None:
        """Reset dynamic states to None."""
        self.cache = None
        self.batch_size = None
        self.current_tokens = None
