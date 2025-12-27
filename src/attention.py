from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from src.rope import RoPE
from src.rms_norm import RMSNorm
from src.kv_cache import KVCache
from gpu_setup import (
    device, 
    dtype, 
    use_amp, 
    flash_attn_kvpacked_func, 
    USE_FLASH_ATTN
)

class CausalAttention(nn.Module):
    """Causal attention layer where optimized attention is utilized if available.
    `qk_norm_type` must be either L2 or RMS normalization, L2 is preferred.
    Gain parameter not included in RMSNorm variant of QK norm. Default
    attention type is GQA. MQA is supported for any device with constraints.
    Flash and windowed attention work with device of CUDA. If windowed attention
    is not supported, left and rights windows are set to -1, enabling global attention.

    Args:
        d_model (int): Dimensionality of model embeddings.
            Must be divisible by `num_heads`.
        num_heads (int): Number of attention heads.
            Must be divisible by `query_groups`.
        query_groups (int): Query groups for GQA.
           `query_groups == 1` and `use_mqa = True` applies MQA.
        softmax_scale (float): Scaler for attention computation.
        rope_theta (float): Exponential base for RoPE inverse frequency.
            inv_freq_i = 1/(rope_theta^(2i/d))
        use_proj_bias (bool): Whether to use projection bias.
        use_qk_norm (bool): Whether to use QK normalization.
        qk_norm_eps (float): Epsilon value to avoid division by zero.
        qk_norm_type (Literal["L2", "RMS"]): Type of QK norm to apply.
            Defaults to L2 norm.
        use_windowed_attn (bool): Whether to use windowed attention.
        use_flash_attn (bool): Whether to use flash attention.
        use_mqa (bool): Whether to use MQA.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        rope_theta: float,
        use_proj_bias: bool = False,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        qk_norm_type: Literal["L2", "RMS"] = "L2",
        use_windowed_attn: bool = True,
        use_flash_attn: bool = True,
        use_mqa: bool = False
    ):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % query_groups == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.qk_norm_type = qk_norm_type
        self.use_windowed_attn = use_windowed_attn and device.type == "cuda"
        self.use_flash_attn = (
            use_flash_attn
            and device.type == "cuda"
            and USE_FLASH_ATTN
            and flash_attn_kvpacked_func is not None
        )
        self.use_mqa = use_mqa
        self.head_dim = d_model // num_heads
        self.num_kv_heads = num_heads // query_groups

        # [d_model, h*d+2*num_kv_heads*d]
        # h*d is for querys, 2*num_kv_heads*d is for KVs split
        self.w_qkv = nn.Linear(
            d_model,
            num_heads*self.head_dim+2*self.num_kv_heads*self.head_dim,
            bias=use_proj_bias,
            device=device,
            dtype=dtype
        )
        self.w_o = nn.Linear(
            d_model, d_model,
            bias=use_proj_bias,
            device=device,
            dtype=dtype
        )

        self.rope = RoPE(self.head_dim, rope_theta)

    def _update_cache(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        kv_cache: KVCache
    ) -> tuple[Tensor, Tensor]:
        """Update cache and get new KV tensors with respect certain layer.
        KV cache is updated with new KV tensors and past KV tensors are
        retrieved. Past KV tensors are concatenated along with new KV tensors.
        K_cache = concat(K_past, K_new), V_cache = concat(V_past, V_new).
        Tensors are concatenated along the sequence length dimension.
        
        Args:
            k (Tensor): Key tensor of shape [b, t, h, d].
            v (Tensor): Value tensor of shape [b, t, h, d].
            layer_idx (int): Current layer to be updated.
            kv_cache (KVCache): KV caching module.
        
        Returns:
            tuple:
                - Tensor: K_cache of shape [b, h, t, d].
                - Tensor: V_cache of shape [b, h, t, d].
        """
        # permute to [b, h, t, d]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        kv_cache.update(k, v, layer_idx)
        past_k, past_v = kv_cache.get(layer_idx)
        if past_k is not None and past_v is not None:
            # concatenate over seqlen dim
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        return k, v

    def _apply_qk_norm(
        self,
        q: Tensor,
        k: Tensor,
        norm_type: Literal["L2", "RMS"] = "L2",
        eps: float = 1e-6
    ) -> tuple[Tensor, Tensor]:
        """Apply QK normalization to input QK tensors.
        L2: sqrt(sum(|x|^2) + eps), RMS: sqrt(1/d * sum(x^2)) + eps)
        
        Args:
            q (Tensor): Query tensor with shape [:, :, :, d].
            k (Tensor): Key tensor with shape [:, :, :, d].
            norm_type (Literal["L2", "RMS"]): Type of normalization to apply.
            eps (float): Epsilon value to avoid division by zero

        Returns:
            tuple:
                - Tensor: Normalized query tensor.
                - Tensor: Normalized value tensor.

        Raises:
            ValueError: norm_type not in ["L2", "RMS"].
        """
        if norm_type == "L2":
            return (
                F.normalize(q, p=2, dim=-1, eps=eps),
                F.normalize(k, p=2, dim=-1, eps=eps)
            )
        elif norm_type == "RMS":
            return (
                F.rms_norm(q, q.size(-1), eps=eps),
                F.rms_norm(k, k.size(-1), eps=eps)
            )
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def _extend_kv_heads(
        self,
        k: Tensor,
        v: Tensor,
        repeats: int,
        dim: int,
        use_mqa: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Extend KV heads before attention computation.
        MQA is used if `query_groups == 1` and `use_mqa` is `True`.

        Args:
            k (Tensor): Key tensor of shape [b, t, num_kv_heads, d].
            v (Tensor): Value tensor of shape [b, t, num_kv_heads, d].
            repeats (int): Repeats to apply over certain dimension.
            dim (int): Dimension to be repeated over.
            use_mqa (bool): Whether to use MQA or not.
                If True and `query_groups == 1`, k and v are returned, unrepeated.
            
        Returns:
            tuple:
                - Tensor: Key tensor of [b, t, h, d].
                - Tensor: Value tensor of [b, t, h, d].
        """
        if use_mqa and k.size(dim) == 1 and v.size(dim) == 1:
            return k, v
        return (
            k.repeat_interleave(repeats, dim=dim),
            v.repeat_interleave(repeats, dim=dim)
        )

    def _setup_qkv(
        self,
        x: Tensor,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Set up QKV tensors for attention.
        KV caching is used if `use_cache` is `True`, `kv_cache` is not `None`,
        and `layer_idx` is not `None`. If the input sequence length is 0,
        three empty tensors (q, k, v) of shape [b, h, t, d] are returned.

        Args:
            x (Tensor): Input tensor of shape [b, t, d_model].
            use_cache (bool): Whether to use KV caching.
            kv_cache (KVCache, optional): KV caching module.
            layer_idx (int, optional): Layer to have KVs updated.

        Returns:
            tuple:
                - Tensor: Query tensor of shape [b, h, t, d].
                - Tensor: Key tensor of shape [b, h, t, d].
                - Tensor: Value tensor of shape [b, h, t, d].
        """
        B, T, _ = x.shape

        # handle 0 input tokens
        if T == 0:
            return (
                torch.empty(
                    B, self.num_heads, T, self.head_dim, 
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B, self.num_heads, T, self.head_dim, 
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B, self.num_heads, T, self.head_dim, 
                    device=x.device, dtype=x.dtype
                )
            )
        
        # get qkv tensors
        # q: [b, t, h*d], k, v: [b, t, h_kv*d]
        qkv = self.w_qkv(x)
        q, kv = torch.split(
            qkv, 
            [self.num_heads*self.head_dim, 
             2*self.num_kv_heads*self.head_dim], 
            dim=-1
        )
        k, v = kv.chunk(2, dim=-1)
        
        assert q.shape == (B, T, self.num_heads*self.head_dim)
        assert k.shape == v.shape == (B, T, self.num_kv_heads*self.head_dim)

        # q: [b, t, h, d], k, v: [b, t, h_kv, d]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_kv_heads, self.head_dim)
        v = v.view(B, T, self.num_kv_heads, self.head_dim)

        # qk norm
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(
                q, k, norm_type=self.qk_norm_type, eps=self.qk_norm_eps
            )

        # rope
        q = self.rope(q)
        k = self.rope(k)

        # extend kv heads
        # flash attn expects k, v in shape [b, t, h_kv, d] before stacking
        if not self.use_flash_attn:
            k, v = self._extend_kv_heads(
                k, v,
                repeats=self.query_groups,
                dim=2,
                use_mqa=self.use_mqa
            ) # [b, t, h, d]

        # update cache if being used
        if use_cache and kv_cache is not None and layer_idx is not None:
            k, v = self._update_cache(k, v, layer_idx, kv_cache)
            return q.permute(0, 2, 1, 3), k, v
        
        # q, k, v: [b, h, t, d]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_flash_attn:
            assert (
                q.shape == 
                k.shape == 
                v.shape == 
                (B, self.num_kv_heads, T, self.head_dim)
            )
        else:
            assert (
                q.shape == 
                k.shape == 
                v.shape == 
                (B, self.num_heads, T, self.head_dim)
            )

        return q, k, v

    def _optimized_attn(
        self,
        x: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        left_window: int = -1,
        right_window: int = -1,
        padding_mask: Tensor | None = None,
        use_causal: bool = True,
        softcap: float = 0.0
    ) -> Tensor:
        """Optimized attention utilizing flash attention and windowed attention.
        Apply `flash_attn_kvpacked_func` to QKV tensors. KV tensors must be packed
        into shape [b, t_k, 2, h_kv, d]. If windowed attention is not being used,
        global attention is used. If causal masking is being used, the right
        window is set to 0. If input sequence length is 0, empty tensor is returned. 
        `flash_attn_kvpacked_func` does not support padding, custom
        padding logic is to used to apply padding.

        Args:
            x (Tensor): Input tensor of shape [b, t, d_model].
            q (Tensor): Query tensor of shape [b, h, t, d].
            k (Tensor): Key tensor of shape [b, h_kv, t, d].
            v (Tensor): Value tensor of shape [b, h_kv, t, d].
            left_window (int): Left window for windowed attention.
                -1 means all previous tokens can be attended to.
            right_window (int): Right window for windowed attention.
                -1 means all future tokens can be attended to.
                0 means no future tokens can be attended to (causal attn).
            padding_mask (optional, Tensor): Padding tensor of shape [B, T_q].
                `True` means compute attention, `False` means mask token.
            use_causal (bool): Whether to apply causal masking.
            softcap (float): Value for softcapped attention.
                0.0 means no softcapping applied.

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        if not self.use_windowed_attn:
            left_window, right_window = -1, -1
        if use_causal:
            right_window = 0
        B, _, T_q, _ = q.shape
        _, _, T_k, _ = k.shape

        # zero input tokens
        if any(tensor.numel() == 0 for tensor in [q, k, v]):
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        # reshape q to [b, t, h, d]
        # reshape k, v to [b, t, h_kv, d] -> kv: [b, t, 2, h_kv, d]
        q, k, v = [tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)]
        kv = torch.stack([k, v], dim=2)

        # TODO: implement padding
        if padding_mask is not None:
            pass
        else:
            out = flash_attn_kvpacked_func(
                q, 
                kv, 
                softmax_scale=self.softmax_scale,
                causal=use_causal,
                window_size=(left_window, right_window),
                softcap=softcap
            ) # [b, t_q, h, d]
        out = out.contiguous().view(B, T_q, -1) # [b, t_q, d_model]

        return self.w_o(out)

    def _dot_product_attn(
        self,
        x: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        padding_mask: Tensor | None = None,
        use_causal: bool = True
    ) -> Tensor:
        """Attention utilizing PyTorch's dot product attention.
        If any of the QKV tensors contain 0 elements (empty tensor),
        another empty tensor will be returned in shape [b, t, d_model].
        Padding mask must have shape [B, T_q]. If padding and causal
        masking is used, a single aggregated mask is created which
        adheres to the rules of causal and padding masking.

        Args:
            x (Tensor): Input tensor of shape [b, t, d_model].
            q (Tensor): Query tensor of shape [b, h, t, d].
            k (Tensor): Key tensor of shape [b, h, t, d.]
            v (Tensor): Value tensor of shape [b, h, t, d].
            padding_mask (optional, Tensor): Padding tensor of shape [b, t_q].
                `True` means compute attention, `False` means mask.
                `None` means no padding masking applied.
            use_causal (bool): Whether to apply causal masking.

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        B, _, T_q, _ = q.shape
        _, _, T_k, _ = k.shape

        # zero input tokens
        if any(tensor.numel() == 0 for tensor in [q, k, v]):
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        # handle padding mask
        if padding_mask is not None:
            assert padding_mask.shape == (B, T_q)
            padding_mask = padding_mask.bool()
            attn_mask = padding_mask[:, None, :, None] # [b, 1, t_q, 1]
            attn_mask = attn_mask.expand(B, 1, T_q, T_k) # [b, 1, t_q, t_k]
            # apply causal masking
            if use_causal:
                # causal_mask: [t_q, t_k]
                causal_mask = torch.tril(
                    torch.ones(T_q, T_k, device=padding_mask.device, dtype=torch.bool)
                )
                causal_mask = causal_mask[None, None, :, :] # [1, 1, t_q, t_k]
                # mask token if it is a pad token or future token to be masked to -inf
                # aggregated mask should follow rules of padding and causality
                attn_mask = attn_mask | causal_mask
        else:
            if use_causal:
                causal_mask = torch.tril(
                    torch.ones(T_q, T_k, device=device, dtype=torch.bool)
                )
                attn_mask = causal_mask[None, None, :, :] # [1, 1, t_q, t_k]
            else:
                attn_mask = None

        # expand to b, h, t_q, t_k
        if attn_mask is not None:
            attn_mask = attn_mask.expand(B, self.num_heads, T_q, T_k)

        # compute attn out
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=use_causal if padding_mask is None else False,
        ) # [b, h, t_q, d]
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1) # [b, t_q, d_model]

        return self.w_o(out)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        *,
        use_causal: bool = True,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        left_window: int = -1,
        right_window: int = -1,
        _return_qkv: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run forward pass of attention module.
        Input tensor must be passed at runtime. Caching is used if 
        `use_cache` is `True`, `kv_cache` is not `None`, `layer_idx` is not `None`.
        Windowed attention is only used in the optimized attention variant.
        In the `padding_mask`, True means compute attention, False means pad.
        `_return_qkv` returns the attention output and QKV tensors for debugging.

        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (Tensor, optional): Padding tensor of shape [B, T_q].
            use_causal (bool): Whether to apply causal masking.
            use_cache (bool): Whether to apply caching.
            kv_cache (KVCache, optional): KV caching module.
            layer_idx (int, optional): Layer index to update KV cache.
            left_window (int): Left window for windowed attention.
                -1 means no model can attend to all past tokens.
            right_window (int): Right window for windowed attention.
                -1 means model can attend to all future tokens.
                0 means model cannot attend to future tokens, used for causal LM.
            _return_qkv (bool): Whether to return QKV tensors for debugging.

        Returns:
            Union:
                - Tensor: Output tensor with same shape as input.
                - tuple: Output tensor and QKV tensors.
        """
        with autocast(device_type=device.type, enabled=use_amp):
            q, k, v = self._setup_qkv(
                x,
                use_cache=use_cache,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )
            if self.use_flash_attn:
                out = self._optimized_attn(
                    x, q, k, v,
                    left_window=left_window,
                    right_window=right_window,
                    padding_mask=padding_mask,
                    use_causal=use_causal
                )
            else:
                out = self._dot_product_attn(
                    x, q, k, v,
                    padding_mask=padding_mask,
                    use_causal=use_causal
                )
            if _return_qkv:
                return out, q, k, v
            return out


class CausalAttentionBlock(nn.Module):
    """Attention block applying normalization, residuals.
    `qk_norm_type` must be either L2 or RMS normalization, L2 is preferred.
    Gain parameter not included in RMSNorm variant of QK norm. Default
    attention type is GQA. MQA is supported for any device with constraints.
    Flash and windowed attention work with device of CUDA. If windowed attention
    is not supported, left and rights windows are set to -1, enabling global attention.
    `dropout_p` is set to `0.0` for evaluation through `model.eval()`.

    Args:
        d_model (int): Dimensionality of model embeddings.
            Must be divisible by `num_heads`.
        num_heads (int): Number of attention heads.
            Must be divisible by `query_groups`.
        query_groups (int): Query groups for GQA.
           `query_groups == 1` and `use_mqa = True` applies MQA.
        softmax_scale (float): Scaler for attention computation.
        rope_theta (float): Exponential base for RoPE inverse frequency.
            inv_freq_i = 1/(rope_theta^(2i/d)).
        dropout_p (float): Dropout probability for regularization.
        rms_norm_eps (float): Epsilon value to avoid division by zero.
        use_proj_bias (bool): Whether to use projection bias.
        use_qk_norm (bool): Whether to use QK normalization.
        qk_norm_eps (float): Epsilon value to avoid division by zero.
        qk_norm_type (Literal["L2", "RMS"]): Type of QK norm to apply.
            Defaults to L2 norm.
        use_windowed_attn (bool): Whether to use windowed attention.
        use_flash_attn (bool): Whether to use flash attention.
        use_mqa (bool): Whether to use MQA.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        softmax_scale: float,
        rope_theta: float,
        dropout_p: float,
        rms_norm_eps: float = 1e-6,
        use_proj_bias: bool = False,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        qk_norm_type: Literal["L2", "RMS"] = "L2",
        use_windowed_attn: bool = True,
        use_flash_attn: bool = True,
        use_mqa: bool = False,
    ):
        super().__init__()

        self.attn = CausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            softmax_scale=softmax_scale,
            rope_theta=rope_theta,
            use_proj_bias=use_proj_bias,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
            qk_norm_type=qk_norm_type,
            use_windowed_attn=use_windowed_attn,
            use_flash_attn=use_flash_attn,
            use_mqa=use_mqa
        )
        self.rms_norm = RMSNorm(
            d_model=d_model, eps=rms_norm_eps
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        *,
        use_causal: bool = True,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        left_window: int = -1,
        right_window: int = -1,
    ) -> Tensor:
        """Run forward pass of attention module.
        Input tensor must be passed at runtime. Caching is used if 
        `use_cache` is `True`, `kv_cache` is not `None`, `layer_idx` is not `None`.
        Windowed attention is only used in the optimized attention variant.
        In the `padding_mask`, True means compute attention, False means pad.

        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            padding_mask (Tensor, optional): Padding tensor of shape [B, T_q].
            use_causal (bool): Whether to apply causal masking.
            use_cache (bool): Whether to apply caching.
            kv_cache (KVCache, optional): KV caching module.
            layer_idx (int, optional): Layer index to update KV cache.
            left_window (int): Left window for windowed attention.
                -1 means no model can attend to all past tokens.
            right_window (int): Right window for windowed attention.
                -1 means model can attend to all future tokens.
                0 means model cannot attend to future tokens, used for causal LM.

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        with autocast(device_type=device.type, enabled=use_amp):
            return x + self.dropout(
                self.attn(
                    self.rms_norm(x),
                    padding_mask=padding_mask,
                    use_causal=use_causal,
                    use_cache=use_cache,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    left_window=left_window,
                    right_window=right_window
                )
            )

def test_attn(grads:bool=True):
    attn = CausalAttention(
        512, 32, 8, 1/(512//32)**0.5, 10000.0
    )
    x = torch.randn(10, 10, 512, device=device, dtype=dtype)
    padding_mask = torch.randint(
        0, 2, (10, 10), device=device, dtype=dtype
    )
    out = attn(x, padding_mask)
    if grads:
        loss = out.sum()
        loss.backward()
        for name, param in attn.named_parameters():
            print(f"{name}: {param.grad}")
    return out

def test_attn_block(grads:bool=True):
    attn = CausalAttentionBlock(
        512, 32, 8, 1/(512//32)**0.5, 10000.0, 0.1
    )
    x = torch.randn(10, 10, 512, device=device, dtype=dtype)
    padding_mask = torch.randint(
        0, 2, (10, 10), device=device, dtype=dtype
    )
    out = attn(x, padding_mask)
    if grads:
        loss = out.sum()
        loss.backward()
        for name, param in attn.named_parameters():
            print(f"{name}: {param.grad}")
    return out
