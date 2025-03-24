import sys
import torch
import torch.utils
import torch.utils.data

from . import transformer_config
from . import training
from . import moe_utils
from . import multi_latent_attention
from . import router
from . import arguments
from . import theoretical_memory_usage

from . import fused_layer_norm


from . import training
def py_patch():
    if sys.version_info >= (3.9, 0):
        return
    import math
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    math.lcm = lcm
    return

# Apply patch
py_patch()

