"""Bounded model-parallel-size-one tests of the actual Megatron/TE patch targets."""

import copy

import pytest
import torch
import torch_musa  # noqa: F401

from musa_titan.kernels import rms_norm_batch_invariant, fused_add_rmsnorm_batch_invariant


def assert_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
        b.detach().contiguous().reshape(-1).view(torch.uint8).cpu(),
    )


@pytest.fixture(scope="module", autouse=True)
def runtime(tmp_path_factory):
    import musa_titan
    musa_titan._apply_runtime_patch()
    from megatron.core import parallel_state
    from musa_titan.megatron import rmsnorm as patch

    torch.distributed.init_process_group(
        backend="gloo", init_method="file://" + str(tmp_path_factory.mktemp("pg") / "init"),
        rank=0, world_size=1,
    )
    parallel_state.initialize_model_parallel()
    patch.apply_patch()
    yield patch
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("zero_centered", (False, True))
def test_real_te_norm_forward_and_backward(runtime, monkeypatch, dtype, zero_centered):
    import transformer_engine.pytorch as te

    module = te.RMSNorm(128, eps=1e-6, device="musa", dtype=dtype, zero_centered_gamma=zero_centered)
    parameter = module.weight
    state_keys = set(module.state_dict())
    with torch.no_grad():
        module.weight.uniform_(-.5, .5)
    x = torch.randn(5, 1, 128, device="musa", dtype=dtype, requires_grad=True)
    calls = []
    original = runtime.rms_norm_batch_invariant

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "rms_norm_batch_invariant", counted)
    y = module(x)
    weight = module.weight + 1 if zero_centered else module.weight
    reference = rms_norm_batch_invariant(x, weight, 1e-6)
    assert calls == [True]
    assert_bytes(y, reference)
    grad = torch.randn_like(y)
    actual_grads = torch.autograd.grad(y, (x, module.weight), grad, retain_graph=True)
    reference_grads = torch.autograd.grad(reference, (x, module.weight), grad)
    for a, b in zip(actual_grads, reference_grads):
        assert_bytes(a, b)
        assert torch.isfinite(a).all() and a.abs().sum() > 0
    assert module.weight is parameter
    assert set(module.state_dict()) == state_keys


def config(**kwargs):
    from megatron.core.transformer.transformer_config import TransformerConfig
    return TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=2, ffn_hidden_size=256,
        normalization="RMSNorm", params_dtype=torch.bfloat16, bf16=True,
        hidden_dropout=0., attention_dropout=0., gradient_accumulation_fusion=False,
        **kwargs,
    )


def test_actual_megatron_layernorm_linear_uses_triton(runtime, monkeypatch):
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    cfg = config()
    module = TELayerNormColumnParallelLinear(
        128, 256, config=cfg, init_method=torch.nn.init.normal_,
        gather_output=False, bias=False, skip_bias_add=False, is_expert=False,
    )
    with torch.no_grad():
        module.weight.mul_(.03)
        module.layer_norm_weight.uniform_(.5, 1.5)
    before = {k: id(v) for k, v in module.named_parameters()}
    calls = []
    original = runtime._rms_norm_forward

    def counted(*args, **kwargs):
        output, rstd = original(*args, **kwargs)
        assert kwargs["out"] is not None
        assert output.data_ptr() == kwargs["out"].data_ptr()
        calls.append(True)
        return output, rstd

    monkeypatch.setattr(runtime, "_rms_norm_forward", counted)
    x = torch.randn(5, 1, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    y, bias = module(x)
    assert bias is None and calls == [True]
    expected = torch.nn.functional.linear(
        rms_norm_batch_invariant(x, module.layer_norm_weight, cfg.layernorm_epsilon),
        module.weight,
    )
    torch.testing.assert_close(y, expected, rtol=.02, atol=.02)
    upstream = torch.randn_like(y)
    parameters = (x, module.weight, module.layer_norm_weight)
    actual_grads = torch.autograd.grad(y, parameters, upstream)
    expected_grads = torch.autograd.grad(expected, parameters, upstream)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
    for grad in actual_grads:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert before == {k: id(v) for k, v in module.named_parameters()}


def test_te_operation_fuser_rmsnorm_uses_triton(runtime, monkeypatch):
    from transformer_engine.pytorch.ops import RMSNorm
    module = RMSNorm(128, eps=1e-6, device="musa", dtype=torch.bfloat16)
    with torch.no_grad():
        module.weight.uniform_(.5, 1.5)
    calls = []
    original = runtime._rms_norm_forward

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "_rms_norm_forward", counted)
    x = torch.randn(3, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    y = module(x)
    assert calls == [True]
    expected = rms_norm_batch_invariant(x, module.weight, module.eps)
    assert_bytes(y, expected)
    upstream = torch.randn_like(y)
    actual_grads = torch.autograd.grad(y, (x, module.weight), upstream)
    expected_grads = torch.autograd.grad(expected, (x, module.weight), upstream)
    for actual, reference in zip(actual_grads, expected_grads):
        assert_bytes(actual, reference)


def test_te_keyword_api_preserves_output_buffer(runtime):
    import transformer_engine_torch as native_te
    from transformer_engine.pytorch import cpp_extensions
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.ops.basic import rmsnorm as op
    assert cpp_extensions.rmsnorm_fwd is native_te.rmsnorm_fwd is op.rmsnorm_fwd
    x = torch.randn(3, 128, device="musa", dtype=torch.bfloat16)
    weight = torch.randn(128, device="musa", dtype=x.dtype)
    output = torch.empty_like(x)
    actual, mu, rstd = cpp_extensions.rmsnorm_fwd(
        input=x, weight=weight, eps=1e-6, ln_out=output, quantizer=None,
        otype=TE_DType[x.dtype], sm_margin=0, zero_centered_gamma=False,
    )
    assert actual is output and mu is None
    assert rstd.dtype == torch.float32 and rstd.shape == (3,)
    assert_bytes(actual, rms_norm_batch_invariant(x, weight, 1e-6))


def make_layer(*, dropout=0.):
    """Use real TransformerLayer scheduling/norms and small attention/MLP modules."""
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

    class Attention(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.proj = torch.nn.Linear(128, 128, bias=False, device="musa", dtype=config.params_dtype)

        def forward(self, x, **kwargs):
            return self.proj(x), None

    class SmallMLP(MLP):
        def __init__(self, config, **kwargs):
            torch.nn.Module.__init__(self)
            self.fc1 = torch.nn.Linear(128, 256, bias=False, device="musa", dtype=config.params_dtype)
            self.fc2 = torch.nn.Linear(256, 128, bias=False, device="musa", dtype=config.params_dtype)

        def forward(self, x):
            return self.fc2(torch.nn.functional.silu(self.fc1(x))), None

    return TransformerLayer(
        config(), TransformerLayerSubmodules(
            input_layernorm=TENorm, self_attention=Attention,
            self_attn_bda=get_bias_dropout_add, pre_mlp_layernorm=TENorm,
            mlp=SmallMLP, mlp_bda=get_bias_dropout_add,
        ), hidden_dropout=dropout,
    )


@pytest.mark.parametrize("checkpoint", (False, True))
def test_real_layer_fuses_add_and_norm_with_gradients(runtime, monkeypatch, checkpoint):
    from megatron.core.tensor_parallel import checkpoint as megatron_checkpoint
    layer = make_layer()
    reference = make_layer()
    reference.load_state_dict(layer.state_dict())
    before = {k: id(v) for k, v in layer.named_parameters()}
    assert set(layer.state_dict()) == set(reference.state_dict())
    calls = []
    original = runtime.fused_add_rmsnorm_batch_invariant

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "fused_add_rmsnorm_batch_invariant", counted)
    x = torch.randn(5, 1, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    function = lambda value: layer(value)[0]
    y = megatron_checkpoint(function, False, x) if checkpoint else function(x)
    assert calls

    # Compose the public kernels with the same learned modules/weights.
    attn, _ = reference.self_attention(reference.input_layernorm(xr))
    n, residual, _, _ = fused_add_rmsnorm_batch_invariant(
        attn, xr, reference.pre_mlp_layernorm.weight, reference.pre_mlp_layernorm.eps,
    )
    mlp, _ = reference.mlp(n)
    expected = mlp + residual
    assert_bytes(y, expected)
    gradient = torch.randn_like(y)
    y.backward(gradient)
    expected.backward(gradient)
    assert_bytes(x.grad, xr.grad)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        layer.named_parameters(), reference.named_parameters(),
    ):
        assert name == ref_name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
        assert_bytes(parameter.grad, ref_parameter.grad)
    if checkpoint:
        assert len(calls) >= 2
    assert before == {k: id(v) for k, v in layer.named_parameters()}
    with torch.no_grad():
        assert torch.isfinite(layer(x.detach())[0]).all()


def test_dropout_and_selective_norm_recompute_keep_native_add(runtime, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported path must not use fused residual RMSNorm")
    monkeypatch.setattr(runtime, "fused_add_rmsnorm_batch_invariant", forbidden)
    layer = make_layer(dropout=.1)
    x = torch.randn(3, 1, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    layer(x)[0].float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    layer = make_layer()
    layer.recompute_pre_mlp_layernorm = True
    # Check the BDA decision directly, without exercising the upstream selective
    # recompute machinery which requires an MLP output-discard hook.
    residual = torch.randn_like(x)
    added = layer.self_attn_bda(True, False)((x, None), residual, 0.)
    assert_bytes(added, x + residual)
    assert not hasattr(added, runtime._CACHE_ATTR)


def test_layer_clone_uses_its_own_norm_weights(runtime):
    layer = make_layer()
    cloned = copy.deepcopy(layer)
    with torch.no_grad():
        cloned.pre_mlp_layernorm.weight.fill_(2.)
    x = torch.randn(3, 1, 128, device="musa", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    added = cloned.self_attn_bda(True, False)((x, None), residual, 0.)
    y = cloned.pre_mlp_layernorm(added)
    expected = fused_add_rmsnorm_batch_invariant(
        x, residual, cloned.pre_mlp_layernorm.weight, cloned.pre_mlp_layernorm.eps,
    )[0]
    assert_bytes(y, expected)
    assert not hasattr(added, runtime._CACHE_ATTR)
    assert torch.equal(layer.pre_mlp_layernorm.weight, torch.ones_like(layer.pre_mlp_layernorm.weight))


def test_attention_only_graph_does_not_cache_across_graph_boundary(runtime, monkeypatch):
    layer = make_layer()
    layer.config.enable_cuda_graph = True
    layer.config.cuda_graph_scope = "attn"

    def forbidden(*args, **kwargs):
        raise AssertionError("attention-only replay must leave the following norm outside fusion")

    monkeypatch.setattr(runtime, "fused_add_rmsnorm_batch_invariant", forbidden)
    x = torch.randn(3, 1, 128, device="musa", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    added = layer.self_attn_bda(True, False)((x, None), residual, 0.)
    assert_bytes(added, x + residual)
    assert not hasattr(added, runtime._CACHE_ATTR)


def test_layer_graph_replay(runtime):
    layer = make_layer().eval()
    x = torch.randn(3, 1, 128, device="musa", dtype=torch.bfloat16)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.no_grad():
        with torch.musa.stream(stream):
            for _ in range(3):
                layer(x)
        stream.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph, stream=stream):
            actual = layer(x)[0]
        for _ in range(2):
            x.normal_()
            expected = layer(x)[0]
            torch.musa.synchronize()
            graph.replay()
            torch.musa.synchronize()
            assert_bytes(actual, expected)


def test_mixed_parameter_dtype_and_local_norm(runtime):
    import transformer_engine.pytorch as te
    from musa_titan.megatron.fused_layer_norm import FusedLayerNorm
    module = te.RMSNorm(128, device="musa", dtype=torch.float32)
    x = torch.randn(3, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    actual = module(x)
    expected = rms_norm_batch_invariant(x.float(), module.weight, module.eps)
    assert_bytes(actual, expected)
    actual.square().mean().backward()
    assert x.grad.dtype == x.dtype and module.weight.grad.dtype == module.weight.dtype
    local = FusedLayerNorm(config(), 128).to("musa")
    actual = local(x)
    expected = rms_norm_batch_invariant(x, local.weight.to(x.dtype), local.eps)
    assert_bytes(actual, expected)
    actual.float().square().mean().backward()
    assert local.weight.grad is not None and torch.isfinite(local.weight.grad).all()


def test_layernorm_falls_back_and_install_is_idempotent(runtime):
    from transformer_engine.pytorch import RMSNorm
    from musa_titan.megatron.fused_layer_norm import FusedLayerNorm
    cfg = config()
    cfg.normalization = "LayerNorm"
    module = FusedLayerNorm(cfg, 128).to("musa")
    x = torch.randn(3, 128, device="musa")
    expected = torch.layer_norm(x, (128,), module.weight, module.bias, module.eps)
    assert_bytes(module(x), expected)
    original = RMSNorm.forward
    runtime.apply_patch()
    assert RMSNorm.forward is original
