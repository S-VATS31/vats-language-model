import os
import torch

_device_type = os.getenv("DEVICE")

if _device_type == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
elif _device_type == "mps" and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
use_amp = device.type == "cuda"

# try to get flash attention function
try:
    from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
    _use_flash_attn = True
except ImportError:
    flash_attn_kvpacked_func = None
    _use_flash_attn = False

USE_FLASH_ATTN = _use_flash_attn and device.dtype == "cuda"
