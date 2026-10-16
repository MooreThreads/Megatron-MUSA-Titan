import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa

from ._version import __version__

_TORCH_PATCHED = False


def register_exit_handler():
    # Clean teardown: drain pending MCCL collectives + stop watchdog threads
    # before interpreter exit (prevents MCCL unhandled system error / SIGABRT race).
    import atexit
    import torch.distributed

    def clean_teardown():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    atexit.register(clean_teardown)


def patch_after_import_torch():
    global _TORCH_PATCHED

    # 1. Patch for torch.xxx
    torch.cuda.is_available = torch.musa.is_available
    # torch.cuda.current_device = lambda : f'musa:{torch.musa.current_device()}'
    torch.cuda.current_device = torch.musa.current_device
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.get_rng_state = torch.musa.get_rng_state
    torch.cuda.set_rng_state = torch.musa.set_rng_state
    torch.cuda.random.get_rng_state = torch.musa.get_rng_state
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.manual_seed = torch.musa.manual_seed
    torch.cuda.Event = torch.musa.Event
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.stream = torch.musa.stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties
    # Activate torch.musa.default_generators before exposing them through CUDA.
    torch.musa.current_device()
    torch.cuda.default_generators = torch.musa.default_generators
    # torch.cuda.amp = torch.musa.amp
    # Memory
    torch.cuda.memory_allocated = torch.musa.memory_allocated
    torch.cuda.max_memory_allocated = torch.musa.max_memory_allocated
    torch.cuda.memory_reserved = torch.musa.memory_reserved
    torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved
    torch.cuda.memory._record_memory_history = torch.musa.memory._record_memory_history
    torch.cuda.memory._snapshot = torch.musa.memory._snapshot

    # (yehua.zhang) replace lazy_call to avoid cpu memory leak,
    # because failure of cuda init in lazy_call will cause endless operation of emplace back.
    torch.cuda._lazy_call = torch.musa.core._lazy_init._lazy_call
    torch.cuda._lazy_init = torch.musa.core._lazy_init._lazy_init

    def get_device_capability_musa(*args, **kwargs):
        # CUDA and MUSA compute capabilities differ. SGLang-MUSA maps CUDA
        # capability values to MUSA values directly, while Megatron remains
        # unmodified and receives this compatibility mapping through a patch.
        from packaging.version import Version
        torch_version = Version(torch.__version__).base_version
        if Version(torch_version) > Version("2.5.0"):
            return (8, 3)
        return [8, 3]
    torch.cuda.get_device_capability = get_device_capability_musa

    if _TORCH_PATCHED:
        return

    # 2.Patch for torch args related to cuda/musa
    def hook_cuda_device(device):
        if isinstance(device, str) and device.startswith("cuda"):
            return device.replace("cuda", "musa")
        if isinstance(device, torch.device) and device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    def maybe_hook_cuda_args(args, kwargs):
        new_args = [hook_cuda_device(arg) for arg in args]
        if "device" in kwargs:
            kwargs["device"] = hook_cuda_device(kwargs["device"])
        else:
            # Preserve the active ``with torch.device(...)`` context after
            # wrapping torch's device constructors below.
            try:
                from torch.utils._device import CURRENT_DEVICE
            except ImportError:
                CURRENT_DEVICE = None
            if CURRENT_DEVICE is not None:
                kwargs["device"] = hook_cuda_device(CURRENT_DEVICE)
        return tuple(new_args), kwargs

    def register_original_device_constructors(originals):
        # DeviceContext dispatches only registered constructor identities.
        # Keep both the original and wrapped functions registered so MUSA
        # contexts do not silently create CPU tensors.
        try:
            from torch.utils._device import _device_constructors
            constructors = _device_constructors()
        except Exception:
            return
        for original in originals:
            constructors.add(original)

    # retain torch.full reference
    original_full = torch.full
    original_device_constructors = [original_full]
    # redeine torch.zeros
    def patched_full(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_full(*args, **kwargs)
        return result
    torch.full = patched_full

    # retain torch.tensor reference
    original_tensor = torch.tensor
    original_device_constructors.append(original_tensor)
    # redefine torch.tensor
    def patched_tensor(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_tensor(*args, **kwargs)
        return result
    torch.tensor = patched_tensor

    # redefine torch.Tensor type
    orig_type = torch.Tensor.type
    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        if isinstance(result, str):
            result = result.replace("musa", "cuda")
        return result
    torch.Tensor.type = musa_type

    # retain torch.zeros reference
    original_zeros = torch.zeros
    original_device_constructors.append(original_zeros)
    # redeine torch.zeros
    def patched_zeros(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_zeros(*args, **kwargs)
        return result
    torch.zeros = patched_zeros

    # retain torch.empty reference
    original_empty = torch.empty
    original_device_constructors.append(original_empty)
    # redifine torch.empty
    def patched_empty(*args, **kwargs):
        args, kwargs = maybe_hook_cuda_args(args, kwargs)
        result = original_empty(*args, **kwargs)
        return result
    torch.empty = patched_empty

    register_original_device_constructors(
        original_device_constructors + [torch.full, torch.tensor, torch.zeros, torch.empty]
    )

    # Original tensor class
    original_is_cuda = torch.Tensor.is_cuda
    def always_cuda(self):
        return self.is_musa

    # TODO : this patch may be override by transformer_engine patch
    # we'd better unify this patch with transformer_engine patch.
    torch.Tensor.is_cuda = property(always_cuda)

    # 3. Patch for nccl/mccl
    origin_init_process_group = torch.distributed.init_process_group
    def patched_init_process_group(*args, **kwargs):
        if 'backend' in kwargs and kwargs['backend'] == 'nccl':
            kwargs['backend'] = 'mccl'
        result = origin_init_process_group(*args, **kwargs)
        return result
    torch.distributed.init_process_group = patched_init_process_group

    _TORCH_PATCHED = True

    # 3. disable pin memory
    # def pin_memory(data, device=None):
    #     return data
    # torch.utils.data._utils.pin_memory.pin_memory = pin_memory

    # 4. disable nvtx
    def _pass_pvtx(*args, **kwargs):
        return
    torch.cuda.nvtx.range_push = _pass_pvtx
    torch.cuda.nvtx.range_pop = _pass_pvtx

    # 5. disable dynamo
    import os
    # os.environ["NVTE_TORCH_COMPILE"] = "0"
    # os.environ["TORCHDYNAMO_DISABLE"] = "1"

    def noop(func=None, *args, **kwargs):                                                                  
        if func is not None:                                                                               
            return func                                                                                    
        return lambda f: f
    # torch.compile = noop
    # torch.jit.script = noop

def py_patch():
    if sys.version_info >= (3.9, 0):
        return
    import math
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    math.lcm = lcm
    return


def patch_sglang_cuda_compatibility():
    """Keep SGLang's MUSA detector from overwriting CUDA compatibility aliases."""
    try:
        from sglang.srt.utils.common import is_musa
    except ModuleNotFoundError as error:
        if error.name != "sglang" and not error.name.startswith("sglang."):
            raise
        return False

    # Newer SGLang only imports torchada from ``is_musa()``.  Importing it
    # here moves its one-time patch application before our compatibility
    # aliases are restored; otherwise the first later ``is_musa()`` call can
    # let torchada overwrite (notably) ``torch.cuda.get_device_capability``.
    setattr(is_musa, "_patched", True)
    is_musa.cache_clear()

    try:
        import torchada  # noqa: F401
    except ModuleNotFoundError as error:
        # torchada is not needed by the old SGLang detector, so preserve the
        # optional integration when running against an older checkout.
        if error.name != "torchada" and not error.name.startswith("torchada."):
            raise

    patch_after_import_torch()

    # ``torchada`` exposes this guard in its private patch module.  Keep it
    # asserted after restoring our aliases so an explicit/lazy apply_patches()
    # call cannot replay torchada's CUDA replacements over musa_titan.
    torchada_patch = sys.modules.get("torchada._patch")
    if torchada_patch is not None:
        torchada_patch._patched = True

    return True

_runtime_patch_applied = False
_megatron_patch_applied = False
_pretrain_patch_applied = False


def _apply_runtime_patch():
    global _runtime_patch_applied
    if _runtime_patch_applied:
        return
    py_patch()
    patch_after_import_torch()
    register_exit_handler()
    _runtime_patch_applied = True


def _apply_megatron_patch():
    global _megatron_patch_applied
    _apply_runtime_patch()
    if _megatron_patch_applied:
        return
    from .megatron import apply_patch as apply_megatron_specific_patch
    apply_megatron_specific_patch()
    _megatron_patch_applied = True


def apply_pretrain_patch():
    """Apply shared Megatron patches and pretraining-specific patches."""
    global _pretrain_patch_applied
    if _pretrain_patch_applied:
        return
    _apply_megatron_patch()
    from .pretrain import apply_patch as apply_pretrain_specific_patch
    apply_pretrain_specific_patch()
    _pretrain_patch_applied = True


def apply_rl_patch():
    """Apply shared Megatron patches and optional slime/miles RL integrations.

    Megatron and SGLang compatibility patches are idempotent. Framework-specific
    slime/miles hooks stay env-gated, so this entry point can be called again
    after ``SLIME_MUSA_PATCH_AT_STARTUP`` or ``MILES_MUSA_PATCH_AT_STARTUP`` is
    set, for example from ``musa_titan.rl.miles_musa.setup()``.
    """
    _apply_megatron_patch()
    # SGLang's ``is_musa()`` installs its own CUDA aliases when it is first
    # evaluated.  Lock that detector before importing RL/SGLang consumers so
    # it cannot replace the compatibility capability shim with the raw MUSA
    # capability (for example, ``(3, 1)``), which makes Megatron reject GroupedGEMM.
    # ``patch_sglang_cuda_compatibility`` is optional and returns False when this
    # patch is used without SGLang.
    patch_sglang_cuda_compatibility()
    from .rl.common import patch_sglang_device_capability
    patch_sglang_device_capability()
    from .rl import apply_patch as apply_rl_specific_patch
    apply_rl_specific_patch()
    patch_after_import_torch()


__all__ = [
    "__version__",
    "apply_pretrain_patch",
    "apply_rl_patch",
    "patch_after_import_torch",
    "patch_sglang_cuda_compatibility",
]
