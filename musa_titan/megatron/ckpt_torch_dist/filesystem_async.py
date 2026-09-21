import logging
from functools import partial
from time import time
from typing import Callable, List, Optional, Tuple, Union

import megatron
import torch
from torch import multiprocessing as mp

from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
    WriteBucket,
)


class FileSystemWriterAsync_patch(FileSystemWriterAsync):
    class _LocalResultsQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    class _OneShotCountQueue:
        def __init__(self, item):
            self.item = item

        def get(self):
            return self.item

        def task_done(self):
            pass

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            1) the function that saves the data.
            2) the function that stages the GPU tensors to a destination for async checkpointing.
                This function should be self-contained.
            3) arguments to that function in 1).
        """
        if not self.write_buckets:
            return None, None, []
        transform_list = [self.transforms] if hasattr(self, "transforms") else []
        return (
            partial(self.write_preloaded_data_singleproc, transform_list, self.use_msc),
            partial(self.preload_tensors, self.write_buckets, True),
            [torch.distributed.get_rank(), self.write_buckets, self.results_queue],
        )

    @staticmethod
    def write_preloaded_data_singleproc(
        transform_list: List,
        use_msc: bool,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
    ) -> None:
        """Write this rank's buckets without spawning per-bucket child processes."""
        logger = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = {}

        for local_proc_idx, write_bucket in enumerate(write_buckets):
            local_results_queue = FileSystemWriterAsync_patch._LocalResultsQueue()
            count_queue = FileSystemWriterAsync_patch._OneShotCountQueue(local_proc_idx)
            try:
                FileSystemWriterAsync.write_preloaded_data(
                    transform_list,
                    local_proc_idx,
                    write_bucket,
                    local_results_queue,
                    count_queue,
                    use_fsync=True,
                    use_msc=use_msc,
                )
                if len(local_results_queue.items) != 1:
                    raise RuntimeError(
                        "Unexpected local write result count "
                        f"{len(local_results_queue.items)} for bucket {local_proc_idx}"
                    )
                result_idx, local_results_or_exc = local_results_queue.items[0]
                if isinstance(local_results_or_exc, Exception):
                    write_results_or_exc = local_results_or_exc
                    break
                write_results_or_exc[result_idx] = local_results_or_exc
            except Exception as exc:
                logger.exception(
                    "FileSystemWriterAsync single-process write failed on rank %s bucket %s",
                    rank,
                    local_proc_idx,
                )
                write_results_or_exc = exc
                break

        global_results_queue.put(write_results_or_exc)
        w_end = time()
        logger.debug(f"{w_end}, rank: {rank}, write(sync,singleproc): {w_end - w_start}")

    @staticmethod
    def preload_tensors(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
        """
        Preloads tensors in `state_dict` to host memory via CPU memory.

        Args:
            write_buckets (List): List of `WriteBucket` objects that define what to
                save in a checkpoint.
            non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
        """

        result = []
        for bucket in write_buckets:
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            tensor_data = [
                (item, tensor.to("cpu", non_blocking=non_blocking)) for item, tensor in tensor_data
            ]
            result.append((file_name, storage_key, (bytes_data, tensor_data)))
        if non_blocking:
            torch.musa.synchronize()
        return result


# patch
megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync = (
    FileSystemWriterAsync_patch
)
