import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa

from ._version import __version__


def register_exit_handler():
    # Clean teardown: drain pending MCCL collectives + stop watchdog threads
    # before interpreter exit (prevents MCCL unhandled system error / SIGABRT race).
    import atexit
    import torch.distributed

    def clean_teardown():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    atexit.register(clean_teardown)


def _get_available_device_type():
    # torch.cuda.is_available is aliased to torch.musa.is_available below, so the
    # CUDA branch would report "cuda" on a MUSA-only machine. PrivateUse1 (musa)
    # must be resolved first.
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return "xpu"
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia"
    # add more available device types here
    return None

def _patch_tensor_type():
    """Install or reuse the CUDA tensor type naming wrapper."""
    orig_type = torch.Tensor.type
    if getattr(orig_type, "_musa_cuda_type_alias", False):
        return

    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        if isinstance(result, str):
            result = result.replace("musa", "cuda")
        return result

    # TE and Titan may both install this wrapper in the same process.
    musa_type._musa_cuda_type_alias = True
    torch.Tensor.type = musa_type


def patch_after_import_torch():
    from torchada import apply_patches

    apply_patches()
    original_device_type = torch._utils._get_available_device_type
    if hasattr(torch.cuda, "_force_cuda_available"):
        torch.cuda._force_cuda_available = True
    torch.cuda.is_available = torch.musa.is_available
    torch._utils._get_available_device_type = _get_available_device_type
    # DCP may have captured the original helper before this initialization.
    # Update that alias without importing DCP or replacing third-party hooks.
    filesystem = sys.modules.get("torch.distributed.checkpoint.filesystem")
    if (
        filesystem is not None
        and getattr(filesystem, "_get_available_device_type", None) is original_device_type
    ):
        filesystem._get_available_device_type = _get_available_device_type
    _patch_tensor_type()

def py_patch():
    # Core 0.19 uses the standard override decorator added in Python 3.12.
    # Keep the installed cp310 TE wheel and use its typing_extensions equivalent.
    import typing
    if not hasattr(typing, "override"):
        from typing_extensions import override
        typing.override = override
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
    # aliases are restored.
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
    # Initialize the SGLang detector before importing RL/SGLang consumers.
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
