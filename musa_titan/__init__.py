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


def _apply_torchada():
    """Install torchada's generic CUDA-to-MUSA compatibility layer."""
    import torchada
    from torchada._patch import _CudaModuleWrapper, apply_patches

    apply_patches()
    # torchada snapshots its availability switch only when it creates the CUDA
    # wrapper.  Megatron may import torchada earlier, so install the MUSA result
    # explicitly and replace any value cached from the first application.
    cuda_module = torch.cuda
    if isinstance(cuda_module, _CudaModuleWrapper):
        cuda_module._force_cuda_available = True
        cuda_module.is_available = torch.musa.is_available
    else:
        torch.cuda.is_available = torch.musa.is_available
    return torchada


def _get_available_device_type():
    """Select the native available device before CUDA compatibility aliases."""
    # torchada makes torch.cuda.is_available() report the MUSA availability.
    # PyTorch's generic selector must therefore check PrivateUse1 first or it
    # will incorrectly select ``cuda`` on a MUSA-only host.
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia"
    return None


def _patch_available_device_type():
    """Patch PyTorch and already-imported DCP device selectors."""
    original_device_type = torch._utils._get_available_device_type
    torch._utils._get_available_device_type = _get_available_device_type

    # DCP may capture the original helper during dependency imports. Update
    # only an alias that still points at PyTorch's original implementation;
    # preserve third-party replacements and avoid eagerly importing DCP.
    filesystem = sys.modules.get("torch.distributed.checkpoint.filesystem")
    if (
        filesystem is not None
        and getattr(filesystem, "_get_available_device_type", None)
        is original_device_type
    ):
        filesystem._get_available_device_type = _get_available_device_type


def _install_musa_specific_cuda_compatibility():
    """Restore behavior that generic torchada redirection does not provide."""
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

    # redefine torch.Tensor type
    orig_type = torch.Tensor.type
    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        if isinstance(result, str):
            result = result.replace("musa", "cuda")
        return result
    torch.Tensor.type = musa_type


def patch_after_import_torch():
    global _TORCH_PATCHED

    _apply_torchada()
    _patch_available_device_type()
    if _TORCH_PATCHED:
        # Another runtime component may import torchada later.  Reapply the
        # values that its generic wrapper must not overwrite.
        _install_musa_specific_cuda_compatibility()
        return

    _install_musa_specific_cuda_compatibility()
    _TORCH_PATCHED = True


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
