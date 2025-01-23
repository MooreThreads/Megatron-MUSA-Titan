import os

from megatron.training.global_vars import (
    get_args,
)

def save_checkpoint(iteration, model, optimizer, opt_param_scheduler):
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
    dlrover_save_checkpoint_dist(iteration, model, optimizer, opt_param_scheduler, 0)
  else:
    dlrover_save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)

def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
  try:
    from dlrover.trainer.torch.flash_checkpoint.megatron_dist_ckpt \
      import load_checkpoint as dlrover_load_checkpoint_dist
    from dlrover.trainer.torch.flash_checkpoint.megatron \
      import load_checkpoint as dlrover_load_checkpoint
  except Exception as e:
    print(f"import flash_ckpt failed {str(e)}")
    return 0

  i = 0
  args = get_args()
  if args.use_distributed_optimizer and not args.no_save_optim:
    i, _ = dlrover_load_checkpoint_dist(model,
                                        optimizer,
                                        opt_param_scheduler,
                                        load_arg,
                                        strict)
  else:
    i = dlrover_load_checkpoint(model,
                                optimizer,
                                opt_param_scheduler,
                                load_arg,
                                strict)

  return i

enable_async_ckpt = int(os.getenv("ENABLE_ASYNC_CKPT", 0))
if enable_async_ckpt:
  import megatron.checkpointing
  megatron.checkpointing.save_checkpoint = save_checkpoint
  megatron.checkpointing.load_checkpoint = load_checkpoint
