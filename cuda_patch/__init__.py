import sys
import torch
import torch.utils
import torch.utils.data
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