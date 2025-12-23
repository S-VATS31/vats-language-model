import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch import Tensor, LongTensor
from torch.utils.checkpoint import checkpoint

from src.kv_cache import KVCache
from src.rms_norm import RMSNorm
from src.block import CausalTransformerBlock
from configs.model_args.args_5M import ModelArgs
from gpu_setup import device, dtype, use_amp

class CausalTransformer(nn.Module):
    """Causal transformer module.
    
    Args:
        model_args (ModelArgs): Model hyperparameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.embedding = nn.Embedding(
            model_args.vocab_size, 
            model_args.d_model,
            device=device,
            dtype=dtype
        )

        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                softmax_scale=model_args.softmax_scale,
                rope_theta=model_args.rope_theta,
                dropout_p=model_args.dropout_p,
                d_ffn=model_args.d_ffn,
                use_mlp_bias=model_args.use_mlp_bias,
                rms_norm_eps=model_args.rms_norm_eps,
                use_proj_bias=model_args.use_proj_bias,
                use_qk_norm=model_args.use_qk_norm,
                qk_norm_eps=model_args.qk_norm_eps,
                qk_norm_type=model_args.qk_norm_type,
                use_windowed_attn=model_args.use_windowed_attn,
                use_flash_attn=model_args.use_flash_attn,
                use_mqa=model_args.use_mqa
            ).to(device) for _ in range(model_args.num_layers)
        ])

        self.kv_cache = KVCache(
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            head_dim=model_args.d_model//model_args.num_heads,
            max_tokens=model_args.max_seq_len
        )

        self.rms_norm = RMSNorm(
            d_model=model_args.d_model, eps=model_args.rms_norm_eps
        )

        self.dropout = nn.Dropout(p=model_args.dropout_p)

        self.lm_head = nn.Linear(
            model_args.d_model,
            model_args.vocab_size,
            bias=False,
            device=device,
            dtype=dtype
        )

        if model_args.use_weight_tying:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Recursively initialize weights for transformer modules."""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: LongTensor,
        padding_mask: Tensor | None = None,
        use_cache: bool = False
    ) -> Tensor:
        """Run forward pass.
        
        Args:
            input_ids (LongTensor): Input IDs of shape [b, t].
            padding_mask (Tensor, optional): Padding tensor of shape [b, t].
            use_cache (bool): Whether to use KV caching.

        Returns:
            Tensor: Logits tensor of shape [b, t, v].
        """
        with autocast(device_type=device.type, enabled=use_amp):
            if input_ids.dtype != torch.int64:
                input_ids = input_ids.long()
            x = self.dropout(self.embedding(input_ids)) # [b, t, d]

            # loop through layers
            for layer_idx, layer in enumerate(self.layers):
                if self.model_args.use_grad_checkpoint:
                    x = checkpoint(
                        layer,
                        x,
                        padding_mask=padding_mask,
                        use_causal=self.model_args.use_causal,
                        use_cache=use_cache,
                        kv_cache=self.kv_cache,
                        layer_idx=layer_idx,
                        left_window=self.model_args.left_window,
                        right_window=self.model_args.right_window,
                        use_reentrant=False
                    )
                else:
                    x = layer(
                        x,
                        padding_mask=padding_mask,
                        use_causal=self.model_args.use_causal,
                        use_cache=use_cache,
                        kv_cache=self.kv_cache,
                        layer_idx=layer_idx,
                        left_window=self.model_args.left_window,
                        right_window=self.model_args.right_window
                    )

            # final rmsnorm
            x = self.rms_norm(x)

            # get logits
            logits = self.lm_head(x) # [b, t, v]

            return logits
        
model = CausalTransformer(ModelArgs()).to(device)
input_ids = torch.randint(
    0, ModelArgs().vocab_size, (10, 15), device=device, dtype=torch.long
)
padding_mask = torch.randint(
    0, 2, (10, 15), device=device, dtype=torch.bool
)
logits = model(input_ids, padding_mask, True)
loss = logits.sum()
loss.backward()
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
print(logits.shape)
print(logits.grad_fn)
print(logits.device)
print(logits.dtype)
