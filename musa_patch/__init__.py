import torch
import torch.utils
import torch.utils.data
import torch_musa
import megatron
import sys


torch.cuda.current_device = lambda : f'musa:{torch.musa.current_device()}'
torch.cuda.device_count = torch.musa.device_count
torch.cuda.set_device = torch.musa.set_device
torch.cuda.DoubleTensor = torch.musa.DoubleTensor
torch.cuda.FloatTensor = torch.musa.FloatTensor
torch.cuda.LongTensor = torch.musa.LongTensor
torch.cuda.HalfTensor = torch.musa.HalfTensor
torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
torch.cuda.IntTensor = torch.musa.IntTensor
torch.cuda.synchronize = torch.musa.synchronize
torch.cuda.get_rng_state = torch.musa.get_rng_state
torch.cuda.set_rng_state = torch.musa.set_rng_state
torch.cuda.synchronize = torch.musa.synchronize
torch.cuda.empty_cache = torch.musa.empty_cache
# torch.cuda.empty = torch.musa.empty
torch.Tensor.cuda = torch.Tensor.musa
torch.cuda.manual_seed = torch.musa.manual_seed
torch.cuda.Event = torch.musa.Event

# Memory
torch.cuda.memory_allocated = torch.musa.memory_allocated
torch.cuda.max_memory_allocated = torch.musa.memory_allocated
torch.cuda.memory_reserved = torch.musa.memory_reserved
torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved
torch.cuda.is_available = torch.musa.is_available
# 保存原始的torch.tensor函数引用
original_tensor = torch.tensor
# 重新定义torch.tensor函数，修改device参数为'cpu'
def patched_tensor(*args, **kwargs):
    if 'device' in kwargs and kwargs['device'] == 'cuda':
        kwargs['device'] = 'musa'
    result = original_tensor(*args, **kwargs)
    return result

# 覆盖原始的torch.tensor
torch.tensor = patched_tensor

# TODO LIST
# device = torch.device('cuda')
# torch.randn
# torch.empty

def norm_forward(self, hidden_states):
    return torch.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.eps)

megatron.core.fusions.fused_layer_norm.FusedLayerNorm.forward = norm_forward

def pin_memory(data, device=None):
    return data
torch.utils.data._utils.pin_memory.pin_memory = pin_memory

orig_type = torch.Tensor.type
def musa_type(*args, **kwargs):
    result = orig_type(*args, **kwargs)
    return result.replace("musa", "cuda")
torch.Tensor.type = musa_type

megatron.legacy.fused_kernels.load = lambda args : None

def set_jit_fusion_options():
    pass

megatron.training.training.set_jit_fusion_options = set_jit_fusion_options
megatron.training.initialize.set_jit_fusion_options = set_jit_fusion_options
# import sys
# for k in sys.modules:
#     if k.startswith('megatron'):
#         if getattr(sys.modules[k], 'set_jit_fusion_options', None):
#             print("wzx debug", str(sys.modules[k]))
#             setattr(sys.modules[k], 'set_jit_fusion_options', set_jit_fusion_options)

