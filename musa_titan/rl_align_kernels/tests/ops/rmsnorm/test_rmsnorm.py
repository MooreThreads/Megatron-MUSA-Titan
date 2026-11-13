"""Common accuracy and invariance matrix, run against both forward operators."""

import pytest
import torch


@pytest.mark.parametrize("hidden", [128, 1024, 2048, 4096])
@pytest.mark.parametrize("rows", [1, 7, 64, 257])
def test_accuracy_regular(case, rows, hidden):
    case.check(case.make((rows, hidden), seed=0))


@pytest.mark.parametrize("hidden", [1023, 1025, 2047, 2049])
@pytest.mark.parametrize("rows", [1, 7])
def test_accuracy_chunk_boundaries(case, rows, hidden):
    case.check(case.make((rows, hidden), seed=1))


def test_invariance_repeat(case):
    inputs = case.make((64, 2048), seed=2)
    expected = case.check(inputs)
    for repeat in range(1, 10):
        case.equal(case.run(inputs), expected, f"repeat={repeat}")


@pytest.mark.parametrize("hidden", [128, 2048])
@pytest.mark.parametrize("rows", [1, 7, 64, 257])
def test_invariance_batch_position_neighbors(case, rows, hidden):
    target_x, target_r, weight = case.make((1, hidden), seed=3)
    expected = case.check((target_x, target_r, weight))
    for pos in sorted({0, rows // 2, rows - 1}):
        for neighbor_seed in (30, 31):
            x, residual, _ = case.make((rows, hidden), seed=neighbor_seed)
            x[pos:pos + 1], residual[pos:pos + 1] = target_x, target_r
            outputs = case.run((x, residual, weight))
            case.equal(tuple(t[pos:pos + 1] for t in outputs), expected,
                       f"rows={rows}, position={pos}, neighbors={neighbor_seed}")


@pytest.mark.parametrize("hidden", [128, 2048])
def test_invariance_padding_permutation_and_chunks(case, hidden):
    x, residual, weight = case.make((7, hidden), seed=3)
    expected = case.check((x, residual, weight))
    padded_x, padded_r, _ = case.make((19, hidden), seed=33)
    positions = torch.tensor([1, 3, 6, 8, 11, 14, 18], device=case.device)
    padded_x[positions], padded_r[positions] = x, residual
    padded = case.run((padded_x, padded_r, weight))
    case.equal(tuple(t[positions] for t in padded), expected, "padding")
    permutation = torch.tensor([6, 1, 4, 0, 5, 2, 3], device=case.device)
    inverse = torch.argsort(permutation)
    permuted = case.run((x[permutation], residual[permutation], weight))
    case.equal(tuple(t[inverse] for t in permuted), expected, "permutation")
    pieces = [case.run((x[a:b], residual[a:b], weight)) for a, b in [(0, 1), (1, 4), (4, 7)]]
    case.equal(tuple(torch.cat(parts, dim=0) for parts in zip(*pieces)), expected, "chunks")


def _strided_copy(tensor):
    storage = torch.empty((*tensor.shape[:-1], tensor.shape[-1] * 2),
                          device=tensor.device, dtype=tensor.dtype)
    view = storage[..., ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def test_invariance_tensor_layout(case):
    x, residual, weight = case.make((6, 2048), seed=4)
    expected = case.check((x, residual, weight))
    shaped = case.check((x.reshape(2, 3, -1), residual.reshape(2, 3, -1), weight))
    case.equal(tuple(a.reshape(b.shape) for a, b in zip(shaped, expected)), expected, "3D")
    # Exercise noncontiguous inputs and weight separately as well as together.
    for selected in [(0,), (1,), (2,), (0, 1, 2)]:
        tensors = tuple(_strided_copy(t) if i in selected else t
                        for i, t in enumerate((x, residual, weight)))
        case.equal(case.check(tensors), expected, f"noncontiguous={selected}")
    single = case.check((x[:1], residual[:1], weight))
    vector = case.check((x[0], residual[0], weight))
    case.equal(tuple(a.reshape(b.shape) for a, b in zip(vector, single)), single, "1D")


@pytest.mark.parametrize("pattern", ["zero", "small", "signed", "dynamic"])
def test_accuracy_numerical_boundaries(case, pattern):
    x, residual, weight = case.make(seed=5)
    if pattern == "zero":
        x.zero_()
        residual.zero_()
    elif pattern == "small":
        x = (x.float() * 1e-4).to(case.dtype)
        residual = (residual.float() * 1e-4).to(case.dtype)
    elif pattern == "dynamic":
        scale = torch.logspace(-4, 4, x.shape[-1]).to(case.device)
        x = (x.float().sign() * scale).to(case.dtype)
        residual = (residual.float().sign() * scale.flip(0)).to(case.dtype)
    case.check((x, residual, weight))


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_accuracy_epsilon(case, eps):
    x, residual, weight = case.make(seed=6)
    x = (x.float() * 1e-4).to(case.dtype)
    residual = (residual.float() * 1e-4).to(case.dtype)
    case.check((x, residual, weight), eps)


@pytest.mark.parametrize("weight_value", [None, 0.0, 1.0, 2.0], ids=["random", "zero", "one", "two"])
def test_accuracy_weight_scaling(case, weight_value):
    x, residual, weight = case.make(seed=7)
    if weight_value is not None:
        weight.fill_(weight_value)
    case.check((x, residual, weight))


@pytest.mark.parametrize("bad_shape", [(128,), (1, 2048)])
def test_contract_bad_weight(case, bad_shape):
    x, residual, _ = case.make(seed=9)
    weight = torch.ones(bad_shape, dtype=case.dtype, device=case.device)
    with pytest.raises(ValueError):
        case.run((x, residual, weight))
