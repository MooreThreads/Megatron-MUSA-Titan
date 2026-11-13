"""Install Titan RMSNorm at Megatron/TE normalization and residual boundaries."""

from functools import partial, wraps
from importlib.metadata import version

import torch

# hack(dinglei)
# Use Titan's batch-invariant forward kernels at the existing normalization boundaries.
from musa_titan.rl_align_kernels.kernels.rmsnorm import fused_add_rmsnorm_batch_invariant, rms_norm_batch_invariant
from musa_titan.rl_align_kernels.kernels.rmsnorm.rmsnorm import _rms_norm_forward
#
from musa_titan.utils.patching import git_commit, replace_attr

_PATCHED = False
_CACHE_ATTR = "_musa_titan_pre_mlp_norm"
_AUTHORED = {
    "author": "dinglei",
    "modified_date": "2026-11-12",
    "upstream": {
        "megatron": {
            "branch": "detached HEAD",
            "commit": "2fe64f17008845c7a71c97af75c2040087783510",
            "commit_date": "2026-07-21T17:09:34+08:00",
        },
        "transformer_engine": {
            "branch": "detached HEAD",
            "commit": "6cbfedca29cd9b051c731649c61471007189445a",
            "commit_date": "2026-06-30T20:38:08+08:00",
        },
    },
}


def _supported(input, weight):
    return (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.ndim > 0
        and input.shape[-1] > 0
        and isinstance(weight, torch.Tensor)
        and weight.device == input.device
        and weight.ndim == 1
        and weight.numel() == input.shape[-1]
        and input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


def _weight(module, dtype):
    weight = module.weight
    if module.zero_centered_gamma:
        weight = weight + 1
    return weight.to(dtype)


def _take_cached(module, input):
    # hack(dinglei)
    # Consume only this norm's fused result once; tensor-local state survives
    # full-layer recomputation without leaving a stale activation on the module.
    cached = getattr(input, _CACHE_ATTR, None)
    if cached is not None and cached[0] is module:
        delattr(input, _CACHE_ATTR)
        return cached[1]
    #
    return None


def apply_patch():
    """Install once before constructing Megatron layers; preserve parameter objects."""
    global _PATCHED
    if _PATCHED:
        return

    import transformer_engine.pytorch as te
    import transformer_engine_torch as native_te
    from transformer_engine.pytorch import cpp_extensions
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.ops.basic import rmsnorm as te_rmsnorm_op
    from transformer_engine.pytorch.ops._common import maybe_autocast_dtype
    from megatron.core.transformer import transformer_layer
    from megatron.core.transformer.identity_op import IdentityOp
    from .fused_layer_norm import FusedLayerNorm

    # A wheel may have no Git metadata. Report its actual installed version,
    # rather than mislabelling the authored checkout as the runtime revision.
    revisions = {
        "megatron": git_commit(transformer_layer.__file__),
        "transformer_engine": "package-version:" + version("transformer_engine"),
    }

    def register(owner, name, replacement, target, reason, aliases=()):
        # hack(dinglei)
        # Update the runtime entry point and explicit aliases together so
        # Megatron/TE actually select Titan kernels; retain patch provenance.
        replace_attr(
            owner, name, replacement, reason=reason, commits=revisions,
            authored=_AUTHORED, upstream_path=target,
            current_path="musa_titan/megatron/rmsnorm.py",
            aliases=aliases, commit_branch_verified=True,
        )
        #

    original_te_forward = te.RMSNorm.forward

    @wraps(original_te_forward)
    def te_forward(self, input, *args, **kwargs):
        # hack(dinglei)
        # Reuse normalization already computed with the preceding residual add.
        cached = _take_cached(self, input)
        if cached is not None:
            return cached
        #
        if args or kwargs or not _supported(input, self.weight) or FP8GlobalStateManager.is_fp8_enabled():
            return original_te_forward(self, input, *args, **kwargs)
        dtype = maybe_autocast_dtype(default_dtype=self.weight.dtype)
        # hack(dinglei)
        # Replace only RMSNorm computation, preserving TE's dtype/gamma handling.
        return rms_norm_batch_invariant(input.to(dtype), _weight(self, dtype), self.eps)
        #

    register(
        te.RMSNorm, "forward", te_forward,
        "transformer_engine.pytorch.RMSNorm.forward",
        "Use Titan RMSNorm for standalone MUSA norms, including Q/K and final norms.",
    )

    original_local_forward = FusedLayerNorm.forward

    @wraps(original_local_forward)
    def local_forward(self, input):
        # hack(dinglei)
        # The local adapter must also consume a fused pre-MLP result only once.
        cached = _take_cached(self, input)
        if cached is not None:
            return cached
        #
        if self.config.normalization != "RMSNorm" or not _supported(input, self.weight):
            return original_local_forward(self, input)
        # hack(dinglei)
        # Route the supported RMSNorm branch to Titan without changing LayerNorm.
        return rms_norm_batch_invariant(input, _weight(self, input.dtype), self.eps)
        #

    register(
        FusedLayerNorm, "forward", local_forward,
        "musa_titan.megatron.fused_layer_norm.FusedLayerNorm.forward",
        "Use Titan RMSNorm in the Megatron local normalization adapter.",
    )

    original_normalize = cpp_extensions.rmsnorm_fwd

    @wraps(original_normalize)
    def rmsnorm_fwd(
        input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma,
    ):
        supported = (
            not zero_centered_gamma
            and quantizer is None and not FP8GlobalStateManager.is_fp8_enabled()
            and _supported(input, weight)
            and otype in (input.dtype, TE_DType[input.dtype])
            and (ln_out is None or (
                isinstance(ln_out, torch.Tensor) and ln_out.shape == input.shape
                and ln_out.is_contiguous() and ln_out.dtype == input.dtype
                and ln_out.device == input.device
            ))
        )
        if not supported:
            return original_normalize(
                input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma,
            )
        # hack(dinglei)
        # TE's fused Linear/MLP autograd still owns GEMM, communication and
        # backward. Supply its expected output buffer and actual forward rstd.
        with torch.no_grad():
            output, rstd = _rms_norm_forward(
                input, weight.to(input.dtype), eps, save_rstd=True, out=ln_out,
            )
        return ln_out if ln_out is not None else output, None, rstd
        #

    register(
        cpp_extensions, "rmsnorm_fwd", rmsnorm_fwd,
        "transformer_engine.pytorch.cpp_extensions.rmsnorm_fwd",
        "Replace TE fused Linear/MLP and operation-fuser RMSNorm without replacing GEMM or backward.",
        aliases=(native_te, te_rmsnorm_op),
    )

    original_init = transformer_layer.TransformerLayer.__init__

    def bda_factory(self, original_factory, training, fused):
        norm = self.pre_mlp_layernorm
        original_bda = original_factory(training, fused)

        def add_and_normalize(output_with_bias, residual, probability):
            x, bias = output_with_bias
            dtype = (
                maybe_autocast_dtype(default_dtype=norm.weight.dtype)
                if isinstance(norm, te.RMSNorm) else x.dtype
            )
            if (
                bias is not None or probability != 0
                or self.recompute_pre_mlp_layernorm
                or FP8GlobalStateManager.is_fp8_enabled()
                or getattr(self.config, "mlp_rms_recompute", False)
                or (
                    (self.config.enable_cuda_graph or self.config.external_cuda_graph)
                    and self.config.cuda_graph_scope not in ("full", "full_iteration")
                )
                or any(getattr(self.config, name, False) for name in (
                    "cpu_offloading", "offload_moe_fc1_input", "offload_moe_fused_swiglu_input",
                    "use_custom_fsdp", "use_torch_fsdp2",
                ))
                or norm._forward_pre_hooks or norm._backward_hooks or norm._backward_pre_hooks
                or not _supported(x, norm.weight)
                or residual.shape != x.shape or residual.device != x.device
                or residual.dtype != x.dtype or dtype != x.dtype
            ):
                return original_bda(output_with_bias, residual, probability)
            # hack(dinglei)
            # Normalize the unrounded FP32 residual sum in one Triton launch;
            # native RMSNorm backward reuses the saved FP32 sum and rstd.
            normalized, added, _, _ = fused_add_rmsnorm_batch_invariant(
                x, residual, _weight(norm, dtype), norm.eps,
            )
            # Decoder cross-attention placeholders preserve this tensor.
            # The next pre-MLP norm consumes the result exactly once.
            # Tensor-local state avoids stale module caches on reentry and
            # full-layer recomputation; the residual remains differentiable.
            setattr(added, _CACHE_ATTR, (norm, normalized))
            return added
            #

        return add_and_normalize

    @wraps(original_init)
    def layer_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        norm = self.pre_mlp_layernorm
        if not (
            self.config.normalization == "RMSNorm"
            and isinstance(norm, (te.RMSNorm, FusedLayerNorm))
            and all(isinstance(m, IdentityOp) for m in (
                self.pre_cross_attn_layernorm, self.cross_attention, self.cross_attn_bda,
            ))
            and not isinstance(self.self_attn_bda, torch.nn.Module)
        ):
            return
        original_factory = self.self_attn_bda

        # hack(dinglei)
        # Replace only the eligible self-attention BDA boundary to fuse the
        # residual add with pre-MLP RMSNorm while retaining Megatron's schedule.
        # partial deep-copies its layer argument, unlike a closure capturing
        # the original layer/norm when actor or critic modules are cloned.
        self.self_attn_bda = partial(bda_factory, self, original_factory)
        #

    register(
        transformer_layer.TransformerLayer, "__init__", layer_init,
        "megatron.core.transformer.transformer_layer.TransformerLayer.__init__",
        "Fuse decoder self-attention residual add with the following standalone RMSNorm.",
    )
    _PATCHED = True
