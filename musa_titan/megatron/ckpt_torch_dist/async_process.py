"""Terminate NVRx checkpoint children when their training rank dies (Linux)."""

import ctypes
import multiprocessing as mp
import os
import signal
from multiprocessing.managers import SyncManager

from musa_titan.utils.patching import replace_attr


def _kill_on_parent_death(parent_pid):
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG: kernel delivery also covers a parent killed with SIGKILL.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The parent may have died before prctl was called.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


try:
    import nvidia_resiliency_ext as nvrx
    from nvidia_resiliency_ext.checkpointing.async_ckpt import core, filesystem_async
except ModuleNotFoundError as error:
    if error.name != "nvidia_resiliency_ext":
        raise
else:
    _original_async_process_target = core.PersistentAsyncCaller.async_process_target

    def async_process_target(*args, **kwargs):
        # multiprocessing retains the creator PID even if it has already exited.
        _kill_on_parent_death(mp.parent_process().pid)
        return _original_async_process_target(*args, **kwargs)

    def get_write_results_queue(mp_mode="spawn"):
        if filesystem_async._results_queue is None:
            manager = SyncManager(ctx=mp.get_context(mp_mode))
            manager.start(initializer=_kill_on_parent_death, initargs=(os.getpid(),))
            filesystem_async._results_queue = manager.Queue()
        return filesystem_async._results_queue

    for owner, name, replacement, upstream_path in (
        (
            core.PersistentAsyncCaller,
            "async_process_target",
            staticmethod(async_process_target),
            "nvidia_resiliency_ext.checkpointing.async_ckpt.core.PersistentAsyncCaller.async_process_target",
        ),
        (
            filesystem_async,
            "get_write_results_queue",
            get_write_results_queue,
            "nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async.get_write_results_queue",
        ),
    ):
        replace_attr(
            owner,
            name,
            replacement,
            reason="Kill checkpoint children when their training rank dies",
            commits={"nvrx": f"installed-package=={nvrx.__version__}"},
            authored={
                "author": "MUSA Titan",
                "modified_date": "2026-10-22",
                "upstream": {},
            },
            upstream_path=upstream_path,
            current_path="musa_titan/megatron/ckpt_torch_dist/async_process.py",
        )
    filesystem_async._get_write_results_queue = get_write_results_queue
