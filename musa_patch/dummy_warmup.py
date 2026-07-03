"""Dummy warmup helpers for pre-training cold-start mitigation."""

import os
from contextlib import contextmanager

import torch
import torch.distributed
from megatron.core import mpu, tensor_parallel
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.training.global_vars import get_args
from megatron.training.training import print_datetime
from megatron.training.utils import print_rank_0, unwrap_model
from transformer_engine.pytorch.cpu_offload import get_fine_grained_offload_handler


def _capture_rng_state():
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
        cuda_rng_tracker = tensor_parallel.get_cuda_rng_tracker()
        if not hasattr(cuda_rng_tracker, "is_initialized") or cuda_rng_tracker.is_initialized():
            state["cuda_rng_tracker"] = cuda_rng_tracker.get_states()
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "cuda_rng_tracker" in state:
        tensor_parallel.get_cuda_rng_tracker().set_states(state["cuda_rng_tracker"])


def _dummy_warmup_loss(output_tensor):
    """Build a scalar loss connected to the model output without training loss side effects."""
    tensors = output_tensor if isinstance(output_tensor, (list, tuple)) else [output_tensor]
    loss = None
    for tensor in tensors:
        if tensor is None or not torch.is_tensor(tensor):
            continue
        tensor_loss = tensor.float().mean()
        loss = tensor_loss if loss is None else loss + tensor_loss

    if loss is None:
        raise RuntimeError("Dummy warmup did not receive a tensor output connected to the model.")
    return loss


def _sp_size():
    args = get_args()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    return tp_size if getattr(args, "sequence_parallel", False) and tp_size > 1 else 1


def _hidden_seq_length():
    args = get_args()
    cp_size = mpu.get_context_parallel_world_size()
    sp_size = _sp_size()
    parallel_seq_size = cp_size * sp_size
    if args.seq_length % parallel_seq_size != 0:
        raise RuntimeError(
            f"Dummy warmup sequence length {args.seq_length} is not divisible by "
            f"CP size {cp_size} and SP size {sp_size}."
        )
    return args.seq_length // parallel_seq_size


def _build_dummy_warmup_batch():
    args = get_args()
    device = torch.cuda.current_device()
    micro_batch_size = args.micro_batch_size
    seq_length = args.seq_length
    vocab_size = max(int(args.padded_vocab_size), 1)

    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(micro_batch_size, seq_length),
        dtype=torch.int64,
        device=device,
    )
    labels = torch.randint(
        low=0,
        high=vocab_size,
        size=(micro_batch_size, seq_length),
        dtype=torch.int64,
        device=device,
    )
    loss_mask = torch.ones((micro_batch_size, seq_length), dtype=torch.float32, device=device)
    position_ids = torch.arange(seq_length, dtype=torch.int64, device=device).unsqueeze(0)
    position_ids = position_ids.expand(micro_batch_size, -1).contiguous()

    if args.create_attention_mask_in_dataloader:
        attention_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device), diagonal=1
        )
        attention_mask = attention_mask.view(1, 1, seq_length, seq_length)
        attention_mask = attention_mask.expand(micro_batch_size, 1, -1, -1).contiguous()
    else:
        attention_mask = None

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # Dummy warmup shapes after slicing:
    #   fake hidden: [S / CP / SP, B, H]
    #   labels/loss_mask/tokens/position_ids: [B, S / CP]
    # Only CP slices targets; SP is gathered back by the LM head before loss.
    batch = get_batch_on_this_cp_rank(batch)
    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
    )


def _build_dummy_warmup_hidden(seq_multiplier=1):
    args = get_args()
    device = torch.cuda.current_device()
    seq_length = _hidden_seq_length() * seq_multiplier
    return torch.randn(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device=device,
        requires_grad=True,
    )


def _vp_rank(model_chunk, chunk_id):
    unwrapped_model = unwrap_model(model_chunk)
    return getattr(unwrapped_model, "vp_stage", chunk_id)


@contextmanager
def _use_vp_rank(model_chunk, chunk_id):
    if mpu.get_virtual_pipeline_model_parallel_world_size() is None:
        yield
        return

    # The real VPP schedule sets the current VP rank before running each local chunk.
    # Dummy warmup bypasses that schedule, so set only the per-PP chunk id here;
    # the physical PP rank is unchanged and still participates in first/last checks.
    old_vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    mpu.set_virtual_pipeline_model_parallel_rank(_vp_rank(model_chunk, chunk_id))
    try:
        yield
    finally:
        mpu.set_virtual_pipeline_model_parallel_rank(old_vp_rank)


@contextmanager
def _disable_dummy_warmup_grad_checks(model):
    saved = []
    for model_chunk in model:
        ddp_config = getattr(model_chunk, "ddp_config", None)
        if ddp_config is None:
            continue
        saved.append((ddp_config, ddp_config.check_for_nan_in_grad, ddp_config.check_for_large_grads))
        ddp_config.check_for_nan_in_grad = False
        ddp_config.check_for_large_grads = False

    try:
        yield
    finally:
        for ddp_config, check_for_nan_in_grad, check_for_large_grads in saved:
            ddp_config.check_for_nan_in_grad = check_for_nan_in_grad
            ddp_config.check_for_large_grads = check_for_large_grads


def _has_mtp_loss_process(model):
    for model_chunk in model:
        if getattr(unwrap_model(model_chunk), "mtp_loss_process", False):
            return True
    return False


def _sync_dummy_warmup_grads(model):
    args = get_args()
    sync_grads_env = os.getenv("DUMMY_WARMUP_SYNC_GRADS")
    if sync_grads_env is not None:
        sync_grads = int(sync_grads_env) != 0
    else:
        sync_grads = not _has_mtp_loss_process(model)
    if not sync_grads:
        return

    with _disable_dummy_warmup_grad_checks(model):
        with torch.profiler.record_function("dummy_warmup_grad_sync"):
            for model_chunk in model:
                if not hasattr(model_chunk, "finish_grad_sync"):
                    continue
                try:
                    model_chunk.finish_grad_sync()
                except AssertionError as error:
                    if (
                        not args.overlap_grad_reduce
                        or not hasattr(model_chunk, "start_grad_sync")
                        or "Communication call has not been issued" not in str(error)
                    ):
                        raise
                    model_chunk.start_grad_sync()
                    model_chunk.finish_grad_sync()


def _dummy_warmup_forward_local(model_chunk):
    unwrapped_model = unwrap_model(model_chunk)
    tokens, labels, loss_mask, attention_mask, position_ids = _build_dummy_warmup_batch()

    pre_process = getattr(unwrapped_model, "pre_process", mpu.is_pipeline_first_stage())
    post_process = getattr(unwrapped_model, "post_process", mpu.is_pipeline_last_stage())
    mtp_process = getattr(unwrapped_model, "mtp_process", False)
    mtp_loss_process = getattr(unwrapped_model, "mtp_loss_process", False)

    if not pre_process:
        # Standalone MTP-loss stages consume concatenated [main, mtp...] hidden states.
        # If this local chunk also owns the MTP layer, the MTP block creates that concat itself.
        seq_multiplier = 1
        if mtp_loss_process and not mtp_process:
            seq_multiplier += int(getattr(get_args(), "mtp_num_layers", 0) or 0)
        unwrapped_model.set_input_tensor(_build_dummy_warmup_hidden(seq_multiplier))
        if not mtp_process:
            tokens = None
            position_ids = None

    if not (post_process or mtp_loss_process):
        labels = None
        loss_mask = None

    return model_chunk(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
    )


def dummy_warmup_step(model, optimizer):
    """Run one dummy microbatch to trigger cold-start kernels without updating weights."""
    args = get_args()
    offload_handler = get_fine_grained_offload_handler()
    offload_handler.num_microbatches = 1

    try:
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        for chunk_id, model_chunk in enumerate(model):
            with _use_vp_rank(model_chunk, chunk_id):
                output_tensor = _dummy_warmup_forward_local(model_chunk)
                _dummy_warmup_loss(output_tensor).backward()

        _sync_dummy_warmup_grads(model)
    finally:
        # Keep warmup invisible to the real training iteration: no optimizer step, no LR step,
        # no iteration/sample accounting, and no leftover gradients after optional grad sync.
        offload_handler.num_microbatches = get_num_microbatches()
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()


def _dummy_warmup_enabled():
    return int(os.getenv("ENABLE_DUMMY_WARMUP", 0)) != 0


def _run_dummy_warmup(model, optimizer):
    rng_state = _capture_rng_state()
    print_datetime('before dummy warmup')
    try:
        with torch.profiler.record_function("dummy_warmup"):
            dummy_warmup_step(model, optimizer)
    finally:
        _restore_rng_state(rng_state)

    # All ranks must leave local dummy work before the real PP schedule starts.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print_datetime('after dummy warmup')


def maybe_run_dummy_warmup(forward_step_func, train_data_iterator, model, optimizer, config, forward_backward_func):
    """Optionally run one local dummy microbatch before the first measured iteration.

    Supported: GPT-like mcore models with pipeline parallelism, including local VPP chunks.
    Not supported here: PP=1, packed/variable sequence paths, and non-GPT model forward
    signatures.
    """
    if not _dummy_warmup_enabled():
        return

    args = get_args()
    if args.pipeline_model_parallel_size == 1:
        print_rank_0("Skipping dummy warmup: pipeline_model_parallel_size=1.")
        return
    if getattr(args, "variable_seq_lengths", False):
        print_rank_0("Skipping dummy warmup: variable sequence lengths are not supported.")
        return
    if getattr(args, "use_legacy_models", False):
        print_rank_0("Skipping dummy warmup: legacy model forward is not supported.")
        return

    _run_dummy_warmup(model, optimizer)
