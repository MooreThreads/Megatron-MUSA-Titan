"""Flash checkpoint integration; preserve the adapter's existing behavior."""

import os

from musa_titan.utils.patching import git_commit, replace_attr

from megatron.training.global_vars import (
    get_args,
)

def save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far,
                    checkpointing_context=None, pipeline_rank=None,
                    expert_rank=None, tensor_rank=None,
                    pipeline_parallel=None, expert_parallel=None,
                    non_persistent_ckpt=False,
                    train_data_iterator=None, ft_client=None,
                    preprocess_common_state_dict_fn=None
                    ):
  try:
    from dlrover.trainer.torch.flash_checkpoint.megatron_dist_ckpt \
      import save_checkpoint as dlrover_save_checkpoint_dist
    from dlrover.trainer.torch.flash_checkpoint.megatron \
      import save_checkpoint as dlrover_save_checkpoint
  except Exception as e:
    print(f"import flash_ckpt failed {str(e)}")
    return

  args = get_args()
  if args.use_distributed_optimizer and not args.no_save_optim:
    dlrover_save_checkpoint_dist(iteration, model, optimizer,
                                 opt_param_scheduler, 0,
                                 preprocess_common_state_dict_fn)
  else:
    dlrover_save_checkpoint(iteration, model, optimizer,
                            opt_param_scheduler, 0)

def load_checkpoint(model, optimizer, opt_param_scheduler,
                    load_arg='load', strict=True,
                    ft_client=None, checkpointing_context=None,
                    skip_load_to_model_and_opt=False):
  try:
    from dlrover.trainer.torch.flash_checkpoint.megatron_dist_ckpt \
      import load_checkpoint as dlrover_load_checkpoint_dist
    from dlrover.trainer.torch.flash_checkpoint.megatron \
      import load_checkpoint as dlrover_load_checkpoint
  except Exception as e:
    print(f"import flash_ckpt failed {str(e)}")
    return 0, 0

  i = 0
  args = get_args()
  if args.use_distributed_optimizer and not args.no_save_optim:
    i,  num_floating_point_operations_so_far = dlrover_load_checkpoint_dist(model,
                                        optimizer,
                                        opt_param_scheduler,
                                        load_arg,
                                        strict)
  else:
    i, num_floating_point_operations_so_far = dlrover_load_checkpoint(model,
                                optimizer,
                                opt_param_scheduler,
                                load_arg,
                                strict,
                                ft_client=ft_client,
                                checkpointing_context=checkpointing_context,
                                skip_load_to_model_and_opt=skip_load_to_model_and_opt)

  return i, num_floating_point_operations_so_far

enable_async_ckpt = int(os.getenv("ENABLE_ASYNC_CKPT", 0))
if enable_async_ckpt:
  print("flash ckpt enabled")
  import megatron.training.checkpointing

  # Upstream target: NVIDIA/Megatron-LM, megatron/training/checkpointing.py.
  # Repository baseline: fdfcef87 (README); per-function compatibility not verified.
  # Change: delegate save/load to dlrover; update the explicit training aliases.
  # Historical provenance is unverified; do not substitute runtime HEAD.
  authored = {
      "author": "unknown", "modified_date": "unknown",
      "upstream": {"megatron": {
          "branch": "unknown", "commit": "unknown", "commit_date": "unknown",
      }},
  }
  replace_attr(megatron.training.checkpointing, "save_checkpoint", save_checkpoint,
          reason="Delegate checkpoint saves to dlrover flash checkpointing",
          authored=authored,
          commits={"megatron": git_commit(megatron.training.checkpointing.__file__)},
          upstream_path="megatron.training.checkpointing.save_checkpoint",
          current_path="musa_titan/megatron/checkpointing.py",
          aliases=(megatron.training.training,))
  replace_attr(megatron.training.checkpointing, "load_checkpoint", load_checkpoint,
          reason="Delegate checkpoint restores to dlrover flash checkpointing",
          authored=authored,
          commits={"megatron": git_commit(megatron.training.checkpointing.__file__)},
          upstream_path="megatron.training.checkpointing.load_checkpoint",
          current_path="musa_titan/megatron/checkpointing.py",
          aliases=(megatron.training.training,))
