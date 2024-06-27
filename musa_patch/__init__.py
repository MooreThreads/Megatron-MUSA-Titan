import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa


def patch_before_import_megatron():
    # Import fused_layer_norm before transformer_engine
    # 因为local/te的last norm均依赖TENorm
    from . import fused_layer_norm
    # Use a fake transformer_engine to disable the actual transformer_engine
    import sys
    from . import transformer_engine
    sys.modules['megatron.core.transformer.custom_layers.transformer_engine'] = transformer_engine

    # Import other necessary modules to patch
    from . import dot_product_attention
    from . import training
    from . import arguments
    from . import linear_with_grad_accumulation_and_async_allreduce

    # Disable some unsupprted features
    # set_jit_fusion_options
    def set_jit_fusion_options():
        pass
    import megatron.training.initialize
    megatron.training.training.set_jit_fusion_options = set_jit_fusion_options
    megatron.training.initialize.set_jit_fusion_options = set_jit_fusion_options
    # Disable fused_kernels
    import megatron.legacy.fused_kernels
    megatron.legacy.fused_kernels.load = lambda args : None
    # Disable _compile_dependencies
    def _compile_dependencies():
        pass
    megatron.training.initialize._compile_dependencies = _compile_dependencies


def patch_after_import_torch():
    # 1. Patch for torch.xxx
    torch.cuda.is_available = torch.musa.is_available
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
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.manual_seed = torch.musa.manual_seed
    torch.cuda.Event = torch.musa.Event
    # Memory
    torch.cuda.memory_allocated = torch.musa.memory_allocated
    torch.cuda.max_memory_allocated = torch.musa.memory_allocated
    torch.cuda.memory_reserved = torch.musa.memory_reserved
    torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved

    # 2.Patch for torch args related to cuda/musa
    # 保存原始的torch.tensor函数引用
    original_tensor = torch.tensor
    # 重新定义torch.tensor
    def patched_tensor(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_tensor(*args, **kwargs)
        return result
    torch.tensor = patched_tensor
    # 重新定义torch.Tensor
    orig_type = torch.Tensor.type
    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        return result.replace("musa", "cuda")
    torch.Tensor.type = musa_type

    # 3. Patch for nccl/mccl
    origin_init_process_group = torch.distributed.init_process_group
    def patched_init_process_group(*args, **kwargs):
        if 'backend' in kwargs and kwargs['backend'] == 'nccl':
            kwargs['backend'] = 'mccl'
        result = origin_init_process_group(*args, **kwargs)
        return result
    torch.distributed.init_process_group = patched_init_process_group

    # 3. disable pin memory
    # def pin_memory(data, device=None):
    #     return data
    # torch.utils.data._utils.pin_memory.pin_memory = pin_memory


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

patch_before_import_megatron()

patch_after_import_torch()


