import math
from typing import Literal
import torch
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """5 million parameter configuration of model arguments."""
    d_model: int = 256
    num_heads: int = 16
    query_groups: int = 2
    d_ffn: int = 1024
    num_layers: int = 8
    dropout_p: float = 0.1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-7
    qk_norm_eps: float = 1e-8
    left_window: int = -1
    right_window: int = 0
    vocab_size: int = 512
    max_seq_len: int = 128
    use_grad_checkpoint: bool = True
    use_proj_bias: bool = False
    use_mqa: bool = False
    softmax_scale: float = 1 / math.sqrt(256//16)
    use_mlp_bias: bool = False
    use_qk_norm: bool = True
    qk_norm_type: Literal["L2", "RMS"] = "L2"
    use_weight_tying: bool = True
    use_causal: bool = True
    use_windowed_attn: bool = True
    use_flash_attn: bool = True
    # generation args
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    use_cache: bool = True
    repetition_penalty: float = 1.7
    return_only_new_tokens: bool = True
    